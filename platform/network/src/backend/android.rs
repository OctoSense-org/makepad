use std::{
    io,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use super::{NetworkBackend, UnsupportedBackend};

pub trait PlatformSocketStream: Send {
    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()>;
    fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()>;
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
    fn write(&mut self, buf: &[u8]) -> io::Result<usize>;
    fn flush(&mut self) -> io::Result<()>;
    fn shutdown(&mut self);
}

pub trait PlatformSocketFactory: Send + Sync {
    fn connect(
        &self,
        host: &str,
        port: &str,
        use_tls: bool,
        ignore_ssl_cert: bool,
    ) -> io::Result<Box<dyn PlatformSocketStream>>;
}

fn backend_slot() -> &'static Mutex<Option<Arc<dyn NetworkBackend>>> {
    static SLOT: OnceLock<Mutex<Option<Arc<dyn NetworkBackend>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn socket_factory_slot() -> &'static Mutex<Option<Arc<dyn PlatformSocketFactory>>> {
    static SLOT: OnceLock<Mutex<Option<Arc<dyn PlatformSocketFactory>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

pub fn register_platform_backend(backend: Arc<dyn NetworkBackend>) {
    if let Ok(mut slot) = backend_slot().lock() {
        *slot = Some(backend);
    }
}

pub fn clear_platform_backend() {
    if let Ok(mut slot) = backend_slot().lock() {
        *slot = None;
    }
}

pub fn register_platform_socket_factory(factory: Arc<dyn PlatformSocketFactory>) {
    if let Ok(mut slot) = socket_factory_slot().lock() {
        *slot = Some(factory);
    }
}

pub fn clear_platform_socket_factory() {
    if let Ok(mut slot) = socket_factory_slot().lock() {
        *slot = None;
    }
}

pub(crate) fn connect_platform_socket_stream(
    host: &str,
    port: &str,
    use_tls: bool,
    ignore_ssl_cert: bool,
) -> io::Result<Box<dyn PlatformSocketStream>> {
    let slot = socket_factory_slot().lock().map_err(|_| {
        io::Error::new(
            io::ErrorKind::Other,
            "android socket stream factory lock poisoned",
        )
    })?;

    match slot.as_ref() {
        Some(factory) => factory.connect(host, port, use_tls, ignore_ssl_cert),
        // OpenHarmony has no platform shim yet: plain TCP through the standard
        // library is enough for the Studio websocket (PlainTcp transport) and
        // any other clear-text stream; TLS stays unsupported until a shim
        // registers one.
        #[cfg(target_env = "ohos")]
        None => std_tcp::connect(host, port, use_tls),
        #[cfg(not(target_env = "ohos"))]
        None => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "android socket stream shim not registered by makepad-platform",
        )),
    }
}

#[cfg(target_env = "ohos")]
mod std_tcp {
    use super::PlatformSocketStream;
    use std::{
        io::{self, Read, Write},
        net::{Shutdown, TcpStream},
        time::Duration,
    };

    struct StdTcp(TcpStream);

    impl PlatformSocketStream for StdTcp {
        fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.0.set_read_timeout(timeout)
        }
        fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            self.0.set_write_timeout(timeout)
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
        fn shutdown(&mut self) {
            let _ = self.0.shutdown(Shutdown::Both);
        }
    }

    pub(super) fn connect(host: &str, port: &str, use_tls: bool) -> io::Result<Box<dyn PlatformSocketStream>> {
        if use_tls {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "TLS socket streams need a registered platform socket factory on OpenHarmony",
            ));
        }
        let stream = TcpStream::connect(format!("{host}:{port}"))?;
        stream.set_nodelay(true)?;
        Ok(Box::new(StdTcp(stream)))
    }
}

pub(crate) fn create_backend() -> Arc<dyn NetworkBackend> {
    if let Ok(slot) = backend_slot().lock() {
        if let Some(backend) = slot.as_ref() {
            return Arc::clone(backend);
        }
    }
    Arc::new(UnsupportedBackend::new(
        "android backend shim not registered by makepad-platform",
    ))
}
