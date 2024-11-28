use embedder_external::{
    axum::{self, response::IntoResponse, Json},
    serde::{self, ser::SerializeStruct, Serialize},
};

use embedder_err::EmbedderAPIError;

use crate::{
    common::{CONCURRENT_LIMITER, MAX_CONCURRENT_REQUESTS},
    helpers::ConcurrencyLimiter,
};

/// The response for the root endpoint.
#[derive(Clone, Debug, Serialize)]
pub struct HealthResponse {
    max: usize,
    current: usize,
}

impl HealthResponse {
    /// Create a new instance of the root response.
    pub fn new() -> Self {
        let current = CONCURRENT_LIMITER
            .get_or_init(ConcurrencyLimiter::<MAX_CONCURRENT_REQUESTS>::new)
            .current();
        Self {
            max: MAX_CONCURRENT_REQUESTS,
            current,
        }
    }

    /// Get the status code for the response.
    ///
    /// If the number of concurrent requests is:
    /// - greater than or equal to the maximum, the status code will be `503 Service Unavailable`.
    /// - greater than or equal to half the maximum, the status code will be `429 Too Many Requests`.
    /// - less than half the maximum, the status code will be `200 OK`.
    pub fn status_code(&self) -> axum::http::StatusCode {
        match self.current {
            current if current >= self.max => axum::http::StatusCode::SERVICE_UNAVAILABLE,
            current if current >= self.max / 2 => axum::http::StatusCode::TOO_MANY_REQUESTS,
            _ => axum::http::StatusCode::OK,
        }
    }
}

impl IntoResponse for HealthResponse {
    fn into_response(self) -> axum::http::Response<axum::body::Body> {
        (self.status_code(), Json(self).into_response()).into_response()
    }
}

/// The health endpoint, for AWS Application Load Balancer health checks.
///
/// If the number of concurrent requests is:
/// - greater than or equal to the maximum, the status code will be `503 Service Unavailable`.
/// - greater than or equal to half the maximum, the status code will be `429 Too Many Requests`.
/// - less than half the maximum, the status code will be `200 OK`.
///
/// This allows AWS to determine if the server is healthy or not, and route traffic
/// accordingly.
///
/// The status code of this endpoint is not strictly following RESTful conventions; rather it was
/// designed to be informative for AWS.
pub async fn health() -> HealthResponse {
    HealthResponse::new()
}
