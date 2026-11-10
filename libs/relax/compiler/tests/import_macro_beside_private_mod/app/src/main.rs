// `use samename::samename;` imports the macro only: the crate's `mod samename` is private, so
// the name binds nothing in the type namespace and `samename::..` in this module is still the
// crate (icu_collections' cpinvlist.rs: `use zerovec::{ZeroVec, ule::AsULE, zerovec};` above
// `#[zerovec::make_varule(CodePointInversionListULE)]`).
mod list {
    use samename::{samename, ule::twice, Thing};

    pub const K: Thing = samename!(3);

    #[samename::make_twin(ListTwin)]
    pub struct List(pub u32);

    pub fn sum() -> u32 {
        K.0 + twice(4) + samename::ule::twice(1) + samename::plus_one!(5)
    }
}
pub use list::ListTwin;

fn main() {
    println!("import beside private mod {} {} {}", list::sum(), list::List(7).0, ListTwin(9).0);
}
