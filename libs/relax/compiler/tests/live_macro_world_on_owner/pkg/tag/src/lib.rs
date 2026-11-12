use proc_macro::{TokenStream, TokenTree};

/// `#[derive(Tag)]` on `struct X`: `impl crate::Tagged for X { const TAG: u32 = <length of "X"> }`.
#[proc_macro_derive(Tag)]
pub fn derive_tag(item: TokenStream) -> TokenStream {
    let mut it = item.into_iter();
    let mut name = String::new();
    while let Some(t) = it.next() {
        if let TokenTree::Ident(i) = &t {
            if i.to_string() == "struct" {
                if let Some(TokenTree::Ident(n)) = it.next() { name = n.to_string(); }
                break;
            }
        }
    }
    format!("impl crate::Tagged for {name} {{ const TAG: u32 = {}; }}", name.len()).parse().unwrap()
}
