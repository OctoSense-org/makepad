use proc_macro::TokenStream;

/// `#[make_twin(Name)] struct ..`: keeps the item and adds `pub struct Name(pub u32);` (the shape
/// of zerovec's `#[make_varule(NameULE)]`, which adds the ULE type beside the struct).
#[proc_macro_attribute]
pub fn make_twin(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut s = item.to_string();
    s.push_str(&format!("\npub struct {}(pub u32);", attr));
    s.parse().unwrap()
}
