//! The Root Endpoint.
//!

use embedder_external::{
    axum::Json,
    serde::{self, ser::SerializeStruct, Serialize},
};

#[cfg(feature = "status")]
use crate::Status;

const NAME: &str = env!("CARGO_PKG_NAME");
const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The response for the root endpoint.
#[derive(Clone, Debug)]
pub struct RootResponse {
    name: &'static str,
    authors: &'static str,
    description: &'static str,
    version: &'static str,
}

impl RootResponse {
    /// Create a new instance of the root response.
    pub fn new() -> Self {
        Self {
            name: NAME,
            authors: AUTHORS,
            description: DESCRIPTION,
            version: VERSION,
        }
    }
}

impl Serialize for RootResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer
            .serialize_struct("RootResponse", if cfg!(feature = "status") { 5 } else { 4 })?;
        state.serialize_field("name", self.name)?;
        state.serialize_field("authors", self.authors)?;
        state.serialize_field("description", self.description)?;
        state.serialize_field("version", self.version)?;

        #[cfg(feature = "status")]
        {
            let status = Status::get();
            state.serialize_field("status", status.as_ref())?;
        }
        state.end()
    }
}

/// The root endpoint, returning the package metadata.
///
/// Typically used for health checks.
pub async fn root() -> Json<RootResponse> {
    Json(RootResponse::new())
}
