//! All error types for the embedder.
//!

mod base;
pub use base::*;

#[cfg(feature = "api")]
mod api;
#[cfg(feature = "api")]
pub use api::*;
