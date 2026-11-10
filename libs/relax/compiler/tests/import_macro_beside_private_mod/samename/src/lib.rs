// The shape of the zerovec crate: a private module named like the crate, an exported
// `macro_rules!` of the same name, an attribute macro re-exported from the derive crate.
mod samename {
    pub struct Thing(pub u32);
}
pub use crate::samename::Thing;
pub use ibp_mac::make_twin;

pub mod ule {
    pub fn twice(x: u32) -> u32 {
        2 * x
    }
}

#[macro_export]
macro_rules! samename {
    ($v:expr) => {
        $crate::Thing($v)
    };
}

#[macro_export]
macro_rules! plus_one {
    ($v:expr) => {
        $v + 1
    };
}
