//! The program's socket factory on OpenHarmony: plain TCP from std. The system has no TLS
//! library for native code, and the ability's TLS sockets are not wired to the shell yet, so a
//! TLS stream is refused with a reason.
use makepad_network::{AndroidSocketStream, AndroidSocketStreamFactory};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::time::Duration;

struct Factory;
struct Stream(TcpStream);

impl AndroidSocketStreamFactory for Factory {
    fn connect(&self, host: &str, port: &str, use_tls: bool, _ignore_ssl_cert: bool) -> io::Result<Box<dyn AndroidSocketStream>> {
        if use_tls {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "TLS streams are not available on OpenHarmony yet"));
        }
        let port: u16 = port.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid port"))?;
        let stream = TcpStream::connect((host, port))?;
        let _ = stream.set_nodelay(true);
        Ok(Box::new(Stream(stream)))
    }
}

impl AndroidSocketStream for Stream {
    fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.0.set_read_timeout(t)
    }
    fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.0.set_write_timeout(t)
    }
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
    fn peer_certificate(&self) -> Option<Vec<u8>> {
        None
    }
    fn shutdown(&mut self) {
        let _ = self.0.shutdown(Shutdown::Both);
    }
}

pub(crate) fn register() {
    makepad_network::register_android_socket_stream_factory_shim(Arc::new(Factory));
}
