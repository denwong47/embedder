//! API specific errors.

use thiserror::Error;

use embedder_external::{
    axum::{self, http::StatusCode, response::IntoResponse, Json},
    clap, serde_json,
};

use crate::EmbedderError;

/// API specific response types.
///
/// These are structs that are used to serialize the response to the client.
pub mod response {
    use embedder_external::{serde::Serialize, serde_json};

    #[derive(Clone, Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ErrorItem {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub location: Option<String>,

        pub message: String,

        pub value: serde_json::Value,
    }

    #[derive(Clone, Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ErrorModel {
        pub title: String,

        pub status: u16,

        #[serde(rename = "detail")]
        pub description: String,

        #[serde(rename = "instance")]
        pub log_reference: Option<String>,

        #[serde(rename = "type")]
        pub uri_reference: Option<String>,

        pub errors: Vec<ErrorItem>,
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EmbedderAPIError {
    #[cfg(feature = "cli")]
    #[error("{0}")]
    ArgsError(#[from] clap::Error),

    #[error("{0}")]
    EmbedderError(#[from] EmbedderError),

    #[error("{0}")]
    IoError(#[from] std::io::Error),

    #[error("{0}")]
    AxumError(#[from] axum::Error),

    #[error("User terminated.")]
    UserTerminated,

    // This is just a demo of how to export `errors`. It is unlikely that we can check
    // each input and return a list of errors like this.
    #[error("Cannot embed some of the inputs.")]
    CannotEmbedInput(Vec<(usize, String, String)>),

    #[error("{0} is not yet implemented.")]
    NotImplemented(String),
}

impl EmbedderAPIError {
    /// Get the status code for the error.
    pub fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }

    /// Get the variant name of the error.
    ///
    /// This works by taking the debug representation of the error and taking only the leading
    /// characters that are valid in a Rust identifier. Anything that is `{` or `(` is considered
    /// the end of the variant name.
    pub fn variant(&self) -> String {
        format!("{:?}", self)
            .chars()
            .take_while(|c| unicode_ident::is_xid_continue(*c))
            .collect()
    }

    /// Convert the error to an API error model.
    pub fn to_error_model(&self) -> response::ErrorModel {
        // TODO Generate the log reference, uri reference, and errors

        response::ErrorModel {
            title: self.variant(),
            status: self.status_code().as_u16(),
            description: self.to_string(),
            log_reference: None,
            uri_reference: None,
            errors: match self {
                Self::CannotEmbedInput(inputs) => inputs
                    .iter()
                    .map(|(idx, message, reason)| response::ErrorItem {
                        location: Some(format!("documents.{}", idx)),
                        message: message.clone(),
                        value: serde_json::json!({
                            "reason": reason,
                        }),
                    })
                    .collect(),
                _ => vec![],
            },
        }
    }
}

impl IntoResponse for EmbedderAPIError {
    fn into_response(self) -> axum::http::Response<axum::body::Body> {
        (
            self.status_code(),
            Json(self.to_error_model()).into_response(),
        )
            .into_response()
    }
}
