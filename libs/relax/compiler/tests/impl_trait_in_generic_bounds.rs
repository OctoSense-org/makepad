// rustc_ast_lowering (lower_generics, ImplTraitContext::Universal): `impl Trait` written in the
// bounds of a fn's own generics is an anonymous type parameter, like one in a parameter type.
// security-framework's `SecTrust::set_trust_ocsp_response<I: Iterator<Item = impl AsRef<[u8]>>>`
// is written this way, and rustls-platform-verifier calls it for every certificate chain on macOS.
// (In a where clause rustc rejects it: E0562 "`impl Trait` is not allowed in bounds".)
fn total<I: Iterator<Item = impl AsRef<[u8]>>>(items: I) -> usize {
    items.map(|item| item.as_ref().len()).sum()
}

struct Trust;
impl Trust {
    fn set<I: Iterator<Item = impl AsRef<[u8]>>>(&self, items: I) -> Vec<Vec<u8>> {
        items.map(|item| item.as_ref().to_vec()).collect()
    }
}

// Both kinds at once: the bound's anonymous parameter and the argument's.
fn mixed<I: Iterator<Item = impl AsRef<[u8]>>>(items: I, extra: impl AsRef<[u8]>) -> usize {
    total(items) + extra.as_ref().len()
}

#[test]
fn impl_trait_in_a_generic_bound_is_an_anonymous_parameter() {
    assert_eq!(total([vec![1u8, 2], vec![3]].iter()), 3);
    assert_eq!(total(["ab".as_bytes(), b"cde"].into_iter()), 5);
    assert_eq!(Trust.set([[9u8, 8], [7, 6]].iter()), vec![vec![9, 8], vec![7, 6]]);
    assert_eq!(mixed([b"ab"].iter(), "xyz"), 5);
    assert_eq!(mixed(std::iter::empty::<Vec<u8>>(), [0u8; 4]), 4);
}
