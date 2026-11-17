import { socket } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { util } from '@kit.ArkTS';
import makepad from 'libmakepad.so';

const DOMAIN = 0xCA11;

/**
 * The app's TLS streams (the shell's `socket.rs`). Native code has no TLS library on this
 * system, so each stream is a TLSSocket here: the system's TLS stack and its CA store.
 * A stream is named by the id the app picked, as a decimal string. Streams are IPv4.
 */
const streams = new Map<string, socket.TLSSocket>();

/** The certificate's DER bytes, whichever encoding the socket hands it in. */
function der(certificate: socket.X509CertRawData): ArrayBuffer {
  let bytes = certificate.data;
  const text = util.TextDecoder.create('utf-8').decodeToString(bytes.subarray(0, Math.min(bytes.length, 32)));
  if (text.startsWith('-----BEGIN')) {
    const pem = util.TextDecoder.create('utf-8').decodeToString(bytes);
    const body = pem.replace(/-----[A-Z ]+-----/g, '').replace(/\s+/g, '');
    bytes = new util.Base64Helper().decodeSync(body);
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

/** `request` is the id, the host, the port and whether to skip certificate validation, one per line. */
export function socketOpen(request: string): void {
  const parts = request.split('\n');
  const id = parts[0];
  const stream = socket.constructTLSSocketInstance();
  streams.set(id, stream);
  const unchecked = parts[3] === '1';
  const local: socket.NetAddress = { address: '0.0.0.0', port: 0 };
  const remote: socket.NetAddress = { address: parts[1], port: Number(parts[2]) };
  const secure: socket.TLSSecureOptions = { protocols: [socket.Protocol.TLSv12, socket.Protocol.TLSv13] };
  const options: socket.TLSConnectOptions = { address: remote, secureOptions: secure, skipRemoteValidation: unchecked };
  stream.bind(local).then(() => {
    // a TLSSocket takes listeners only once it is bound
    stream.on('message', (info: socket.SocketMessageInfo) => makepad.socketData(id, info.message));
    stream.on('close', () => makepad.socketClosed(id));
    stream.on('error', () => makepad.socketClosed(id));
    return stream.connect(options);
  }).then(() => {
    if (streams.get(id) !== stream) {
      // the app gave up on it meanwhile
      stream.close().catch((_: BusinessError) => {});
      return;
    }
    // The system checked the certificate's chain. The shell checks that it is for this host,
    // so a stream without its certificate only opens when validation is off.
    stream.getRemoteCertificate().then((certificate: socket.X509CertRawData) => {
      makepad.socketOpened(id, true, der(certificate));
    }).catch((_: BusinessError) => makepad.socketOpened(id, true, undefined));
  }).catch((e: BusinessError) => {
    hilog.warn(DOMAIN, 'makepad', 'socket %{public}s did not open: %{public}d %{public}s', id, e.code, e.message);
    makepad.socketOpened(id, false, undefined);
  });
}

export function socketSend(id: string, data: ArrayBuffer | undefined): void {
  const stream = streams.get(id);
  if (stream === undefined || data === undefined) {
    makepad.socketSent(id, false);
    return;
  }
  stream.send(data).then(() => makepad.socketSent(id, true)).catch((e: BusinessError) => {
    hilog.warn(DOMAIN, 'makepad', 'socket %{public}s send: %{public}d %{public}s', id, e.code, e.message);
    makepad.socketSent(id, false);
  });
}

export function socketClose(id: string): void {
  const stream = streams.get(id);
  if (stream === undefined) {
    return;
  }
  streams.delete(id);
  stream.off('message');
  stream.off('close');
  stream.off('error');
  stream.close().catch((_: BusinessError) => {});
}
