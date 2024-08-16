//! API specific errors.

use thiserror::Error;

use crate::EmbedderError;

use axum::{http::StatusCode, response::IntoResponse};

#[derive(Error, Debug)]
#[cfg(features = "api")]
#[non_exhaustive]
pub enum EmbedderAPIError {
    #[error("{0}")]
    EmbedderError(#[from] EmbedderError),
}

impl EmbedderAPIError {
    /// Get the status code for the error.
    fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

impl IntoResponse for EmbedderAPIError {
    fn into_response(self) -> impl IntoResponse {
        ()
    }
}
