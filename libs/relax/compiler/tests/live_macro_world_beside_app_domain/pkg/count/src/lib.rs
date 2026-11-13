use proc_macro::{TokenStream, TokenTree};

/// A static of the macro world with an image that is not zeros: laid out when the derive first
/// runs, which after a live edit is while the program runs.
static TABLE: [u64; 8192] = [0x2222_2222_2222_2222; 8192];

/// `#[derive(Count)]` on `struct X`: `impl X { pub fn count() -> u32 { 7 } }`.
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
    let seven = std::hint::black_box(&TABLE).iter().take(7).filter(|w| **w == 0x2222_2222_2222_2222).count();
    format!("impl {name} {{ pub fn count() -> u32 {{ {seven} }} }}").parse().unwrap()
}
