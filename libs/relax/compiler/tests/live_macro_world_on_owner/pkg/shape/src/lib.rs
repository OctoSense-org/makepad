//! A library of the macro world (the proc-macro crate `count` depends on it) that derives with
//! another proc-macro crate: its files need a macro while the world is being built.
pub trait Tagged {
    const TAG: u32;
}

#[derive(relax_live_world_tag::Tag)]
pub struct Seven77;

mod more;

pub fn base() -> u32 {
    <Seven77 as Tagged>::TAG * 100 + more::extra()
}
