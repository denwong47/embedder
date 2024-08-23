use embedder_external::{axum, serde::Serialize, serde_json};

use embedder_err::EmbedderAPIError;

/// A trait to convert a type to a JSON response.
///
/// This trait is implemented for any type that implements `serde::Serialize`.
pub trait ToJsonResponse {
    fn to_json_response(&self) -> Result<axum::Json<serde_json::Value>, EmbedderAPIError>;
}

impl<T> ToJsonResponse for T
where
    T: Serialize,
{
    fn to_json_response(&self) -> Result<axum::Json<serde_json::Value>, EmbedderAPIError> {
        serde_json::to_value(self)
            .map_err(EmbedderAPIError::from)
            .map(axum::Json)
    }
}

/// Placeholder function to calculate the default batch size.
pub fn calculate_default_batch_size(_count: usize) -> usize {
    16
}
