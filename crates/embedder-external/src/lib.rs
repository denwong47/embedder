#![allow(unused_imports)]
//! This crate does nothing apart from re-exporting all the shared external crates
//! that are used by the other crates in this workspace.
//!
//! This helps synchronizing the versions of the external crates.

#[cfg(feature = "axum")]
pub use axum;

#[cfg(feature = "clap")]
pub use clap;

pub use fastembed;
pub use ndarray;
pub use serde;
pub use serde_json;
pub use utoipa;
