//! Base Error type for the embedder.
//!

use std::boxed::Box;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("Model could not be loaded from {path}: {error}")]
    ModelPathError {
        path: Box<std::path::Path>,
        error: std::io::Error,
    },
    #[error("No input provided.")]
    EmptyInputError,
    #[error("Failed to generate embeddings: {0}")]
    FastEmbedError(#[from] fastembed::Error),
    #[error("Failed to load '{name}': {error}")]
    ModelLoadError { name: &'static str, error: String },
    #[error("Filed to transform the output as desired: {0}")]
    OutputTransformError(String),
    #[error("Failed to parse environment variable '{key}': {error}")]
    EnvVarError { key: String, error: String },
}
