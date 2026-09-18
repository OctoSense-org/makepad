//! The Linux backend's socket stream for OpenHarmony: plain TCP only. The
//! OHOS sysroot ships no OpenSSL, so TLS connections report `Unsupported`;
//! loopback asset servers and plain `http://` / `ws://` endpoints work.
use std::{
    io,
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    time::Duration,
};

pub(crate) enum SocketStream {
    Plain(TcpStream),
}

fn tls_unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "TLS is not available on OpenHarmony (no OpenSSL in the sysroot)",
    )
}

impl SocketStream {
    pub fn connect(
        host: &str,
        port: &str,
        use_tls: bool,
        _ignore_ssl_cert: bool,
    ) -> io::Result<Self> {
        if use_tls {
            return Err(tls_unsupported());
        }
        let tcp_stream = TcpStream::connect(format!("{host}:{port}"))?;
        let _ = tcp_stream.set_nodelay(true);
        Ok(SocketStream::Plain(tcp_stream))
    }

    #[allow(dead_code)]
    pub fn into_tls(self, _host: &str, _ignore_ssl_cert: bool) -> io::Result<Self> {
        Err(tls_unsupported())
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            SocketStream::Plain(stream) => stream.set_read_timeout(timeout),
        }
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            SocketStream::Plain(stream) => stream.set_write_timeout(timeout),
        }
    }

    pub fn shutdown(&mut self) {
        match self {
            SocketStream::Plain(stream) => {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
    }
}

impl Read for SocketStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            SocketStream::Plain(stream) => stream.read(buf),
        }
    }
}

impl Write for SocketStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            SocketStream::Plain(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            SocketStream::Plain(stream) => stream.flush(),
        }
    }
}
