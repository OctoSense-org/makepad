//! Whether a server's certificate is for the host that was dialled (RFC 6125): the names in
//! its subject alternative name extension, read straight from the DER. OpenHarmony's TLS
//! socket checks the chain but lets `*.example.com` stand for `a.b.example.com`, so its shell
//! checks the name itself.

/// One DER element at the front of `bytes`: its tag, its content, and what follows it.
fn element(bytes: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = bytes.split_first()?;
    let (&first, rest) = rest.split_first()?;
    let (len, rest) = if first < 0x80 { (first as usize, rest) } else {
        let count = (first & 0x7f) as usize;
        if count == 0 || count > 4 || rest.len() < count { return None; }
        (rest[..count].iter().fold(0usize, |len, &b| len << 8 | b as usize), &rest[count..])
    };
    if rest.len() < len { return None; }
    Some((tag, &rest[..len], &rest[len..]))
}
/// The content of the element with `tag` at the front of `bytes`, and what follows it.
fn expect(bytes: &[u8], tag: u8) -> Option<(&[u8], &[u8])> {
    let (found, content, rest) = element(bytes)?;
    (found == tag).then_some((content, rest))
}

const SEQUENCE: u8 = 0x30;
/// 2.5.29.17, subjectAltName
const SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];

/// The certificate's alternative names: its DNS names, and its IP addresses as their bytes.
fn names(der: &[u8]) -> Option<(Vec<&str>, Vec<&[u8]>)> {
    let (certificate, _) = expect(der, SEQUENCE)?;
    let (mut tbs, _) = expect(certificate, SEQUENCE)?;
    // version [0], serial, signature, issuer, validity, subject, key, then [1], [2] and the
    // extensions in [3]
    let mut extensions = None;
    while let Some((tag, content, rest)) = element(tbs) {
        if tag == 0xa3 { extensions = Some(content); }
        tbs = rest;
    }
    let (mut list, _) = expect(extensions?, SEQUENCE)?;
    let (mut dns, mut ips) = (Vec::new(), Vec::new());
    while let Some((extension, rest)) = expect(list, SEQUENCE) {
        list = rest;
        let Some((id, mut value)) = expect(extension, 0x06) else { continue };
        if id != SUBJECT_ALT_NAME { continue; }
        // (the critical flag)
        if let Some((_, rest)) = expect(value, 0x01) { value = rest; }
        let (octets, _) = expect(value, 0x04)?;
        let (mut general, _) = expect(octets, SEQUENCE)?;
        while let Some((tag, content, rest)) = element(general) {
            general = rest;
            match tag {
                // dNSName [2], iPAddress [7]
                0x82 => if let Ok(name) = std::str::from_utf8(content) { dns.push(name); },
                0x87 => ips.push(content),
                _ => {}
            }
        }
    }
    Some((dns, ips))
}

/// `pattern` is a certificate's DNS name: `host` itself, or `*.` and all of `host` after its
/// first label.
fn dns_matches(pattern: &str, host: &str) -> bool {
    let pattern = pattern.strip_suffix('.').unwrap_or(pattern);
    if pattern.is_empty() { return false; }
    match pattern.strip_prefix("*.") {
        // a wildcard stands for one whole label, and not for a name right under a top-level one
        Some(parent) => !parent.contains('*') && parent.contains('.')
            && host.split_once('.').is_some_and(|(label, rest)| !label.is_empty() && rest.eq_ignore_ascii_case(parent)),
        None => !pattern.contains('*') && pattern.eq_ignore_ascii_case(host),
    }
}

/// Whether the certificate `der` names `host` (a DNS name or an IP address literal).
pub(crate) fn certificate_names_host(der: &[u8], host: &str) -> bool {
    let Some((dns, ips)) = names(der) else { return false };
    let host = host.strip_suffix('.').unwrap_or(host);
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ips.iter().any(|found| *found == ip.octets().as_slice()),
        Ok(std::net::IpAddr::V6(ip)) => ips.iter().any(|found| *found == ip.octets().as_slice()),
        Err(_) => !host.is_empty() && dns.iter().any(|pattern| dns_matches(pattern, host)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        match content.len() {
            len @ 0..=0x7f => out.push(len as u8),
            len @ 0x80..=0xff => out.extend([0x81, len as u8]),
            len => out.extend([0x82, (len >> 8) as u8, len as u8]),
        }
        out.extend(content);
        out
    }
    /// A certificate's shape with these alternative names and nothing an identity check reads.
    fn certificate(dns: &[&str], ips: &[&[u8]]) -> Vec<u8> {
        let mut general = Vec::new();
        for name in dns { general.extend(tlv(0x82, name.as_bytes())); }
        for ip in ips { general.extend(tlv(0x87, ip)); }
        let san = [tlv(0x06, SUBJECT_ALT_NAME), tlv(0x04, &tlv(SEQUENCE, &general))].concat();
        // an extension before it, with the critical flag
        let other = [tlv(0x06, &[0x55, 0x1d, 0x0f]), tlv(0x01, &[0xff]), tlv(0x04, &[0x03, 0x02, 0x05, 0xa0])].concat();
        let extensions = tlv(0xa3, &tlv(SEQUENCE, &[tlv(SEQUENCE, &other), tlv(SEQUENCE, &san)].concat()));
        let tbs = [
            tlv(0xa0, &tlv(0x02, &[2])), tlv(0x02, &[1; 9]), tlv(SEQUENCE, &[]), tlv(SEQUENCE, &[]),
            tlv(SEQUENCE, &[]), tlv(SEQUENCE, &vec![0; 300]), tlv(SEQUENCE, &[]), extensions,
        ].concat();
        tlv(SEQUENCE, &[tlv(SEQUENCE, &tbs), tlv(SEQUENCE, &[]), tlv(0x03, &[0; 72])].concat())
    }

    #[test]
    fn a_name_matches_itself_only() {
        let der = certificate(&["example.com", "www.example.com"], &[]);
        assert!(certificate_names_host(&der, "example.com"));
        assert!(certificate_names_host(&der, "WWW.Example.COM"));
        assert!(certificate_names_host(&der, "example.com."));
        assert!(!certificate_names_host(&der, "mail.example.com"));
        assert!(!certificate_names_host(&der, "example.org"));
        assert!(!certificate_names_host(&der, "xexample.com"));
        assert!(!certificate_names_host(&der, ""));
    }

    #[test]
    fn a_wildcard_is_one_label() {
        let der = certificate(&["*.badssl.com", "badssl.com"], &[]);
        assert!(certificate_names_host(&der, "expired.badssl.com"));
        assert!(certificate_names_host(&der, "badssl.com"));
        assert!(!certificate_names_host(&der, "wrong.host.badssl.com"));
        assert!(!certificate_names_host(&der, ".badssl.com"));
        assert!(!certificate_names_host(&der, "badssl.com.evil.org"));
    }

    #[test]
    fn odd_wildcards_match_nothing() {
        let der = certificate(&["*.com", "*", "w*.example.com", "*.*.example.com", "a.*.example.com"], &[]);
        for host in ["example.com", "com", "www.example.com", "a.b.example.com", "*", "*.com"] {
            assert!(!certificate_names_host(&der, host), "{host}");
        }
    }

    #[test]
    fn an_address_matches_address_names_only() {
        let v6 = "2606:4700:4700::1111".parse::<std::net::Ipv6Addr>().unwrap().octets();
        let der = certificate(&["one.one.one.one", "1.0.0.1"], &[&[1, 1, 1, 1], &v6]);
        assert!(certificate_names_host(&der, "1.1.1.1"));
        assert!(certificate_names_host(&der, "2606:4700:4700::1111"));
        assert!(certificate_names_host(&der, "one.one.one.one"));
        // a DNS name that reads as an address is not an address name
        assert!(!certificate_names_host(&der, "1.0.0.1"));
        assert!(!certificate_names_host(&der, "8.8.8.8"));
    }

    #[test]
    fn no_names_and_broken_bytes_match_nothing() {
        assert!(!certificate_names_host(&certificate(&[], &[]), "example.com"));
        let der = certificate(&["example.com"], &[]);
        for cut in 0..der.len() { assert!(!certificate_names_host(&der[..cut], "example.com"), "cut at {cut}"); }
        assert!(!certificate_names_host(&[], "example.com"));
        assert!(!certificate_names_host(&[0x30, 0x84, 0xff, 0xff, 0xff, 0xff], "example.com"));
    }
}
