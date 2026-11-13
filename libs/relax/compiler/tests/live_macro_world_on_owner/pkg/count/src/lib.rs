use proc_macro::{TokenStream, TokenTree};

/// `#[derive(Count)]` on `struct X`: `impl X { pub fn count() -> u32 { <shape's base> } }`.
#[proc_macro_derive(Count)]
pub fn derive_count(item: TokenStream) -> TokenStream {
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
    format!("impl {name} {{ pub fn count() -> u32 {{ {} }} }}", relax_live_world_shape::base()).parse().unwrap()
}
