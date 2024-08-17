//! Declaration of all the models supported by the library.

mod sentence_transformers;
pub use sentence_transformers::*;

/// Embedded models for the transformers.
mod binaries {
    include!(concat!(
        env!("OUT_DIR"),
        "/src/transform/models/binaries.rs"
    ));
}
