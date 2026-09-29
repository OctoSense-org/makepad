//! The Linux backend's socket stream for OpenHarmony. The OHOS sysroot ships
//! no OpenSSL for apps, so TLS is rustls (ring provider) with the bundled
//! Mozilla roots from webpki-roots: the same stack the July OHOS port's
//! reqwest backend used on a Mate 70. Plain TCP is unchanged.
use std::{
    io,
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    sync::{Arc, OnceLock},
    time::Duration,
};

use rustls::{
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, ServerName, UnixTime},
    ClientConfig, ClientConnection, DigitallySignedStruct, RootCertStore, SignatureScheme,
    StreamOwned,
};

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn verified_config() -> io::Result<Arc<ClientConfig>> {
    static CONFIG: OnceLock<Result<Arc<ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let roots = RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
            };
            ClientConfig::builder_with_provider(provider())
                .with_safe_default_protocol_versions()
                .map(|b| Arc::new(b.with_root_certificates(roots).with_no_client_auth()))
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(io::Error::other)
}

/// `ignore_ssl_cert`: the request asked to skip verification (a dev server
/// with a self-signed certificate), as the OpenSSL backend allows.
#[derive(Debug)]
struct NoVerify(Arc<rustls::crypto::CryptoProvider>);

impl ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn unverified_config() -> io::Result<Arc<ClientConfig>> {
    static CONFIG: OnceLock<Result<Arc<ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let provider = provider();
            ClientConfig::builder_with_provider(provider.clone())
                .with_safe_default_protocol_versions()
                .map(|b| {
                    Arc::new(
                        b.dangerous()
                            .with_custom_certificate_verifier(Arc::new(NoVerify(provider)))
                            .with_no_client_auth(),
                    )
                })
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(io::Error::other)
}

fn tls_connect(
    tcp_stream: TcpStream,
    host: &str,
    ignore_ssl_cert: bool,
) -> io::Result<StreamOwned<ClientConnection, TcpStream>> {
    let config = if ignore_ssl_cert {
        unverified_config()?
    } else {
        verified_config()?
    };
    let name = ServerName::try_from(host.to_string())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let conn = ClientConnection::new(config, name).map_err(io::Error::other)?;
    let mut stream = StreamOwned::new(conn, tcp_stream);
    // Handshake now, so a certificate or protocol failure is the connect
    // error rather than the first read's.
    while stream.conn.is_handshaking() {
        stream.conn.complete_io(&mut stream.sock)?;
    }
    Ok(stream)
}

pub(crate) enum SocketStream {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
}

impl SocketStream {
    pub fn connect(
        host: &str,
        port: &str,
        use_tls: bool,
        ignore_ssl_cert: bool,
    ) -> io::Result<Self> {
        let tcp_stream = TcpStream::connect(format!("{host}:{port}"))?;
        let _ = tcp_stream.set_nodelay(true);
        if use_tls {
            Ok(SocketStream::Tls(Box::new(tls_connect(
                tcp_stream,
                host,
                ignore_ssl_cert,
            )?)))
        } else {
            Ok(SocketStream::Plain(tcp_stream))
        }
    }

    #[allow(dead_code)]
    pub fn into_tls(self, host: &str, ignore_ssl_cert: bool) -> io::Result<Self> {
        match self {
            SocketStream::Tls(stream) => Ok(SocketStream::Tls(stream)),
            SocketStream::Plain(tcp_stream) => Ok(SocketStream::Tls(Box::new(tls_connect(
                tcp_stream,
                host,
                ignore_ssl_cert,
            )?))),
        }
    }

    fn tcp(&self) -> &TcpStream {
        match self {
            SocketStream::Plain(stream) => stream,
            SocketStream::Tls(stream) => &stream.sock,
        }
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.tcp().set_read_timeout(timeout)
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.tcp().set_write_timeout(timeout)
    }

    pub fn shutdown(&mut self) {
        if let SocketStream::Tls(stream) = self {
            stream.conn.send_close_notify();
            let _ = stream.conn.complete_io(&mut stream.sock);
        }
        let _ = self.tcp().shutdown(Shutdown::Both);
    }
}

impl Read for SocketStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            SocketStream::Plain(stream) => stream.read(buf),
            SocketStream::Tls(stream) => match stream.read(buf) {
                // A peer that closes without close_notify (common for HTTP/1.1
                // servers that end the body by closing) is end of stream.
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(0),
                other => other,
            },
        }
    }
}

impl Write for SocketStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            SocketStream::Plain(stream) => stream.write(buf),
            SocketStream::Tls(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            SocketStream::Plain(stream) => stream.flush(),
            SocketStream::Tls(stream) => stream.flush(),
        }
    }
}
