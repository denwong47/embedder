//! The main `embed` endpoint, converting `documents` into `embeddings`.

use embedder_err::EmbedderAPIError;
use embedder_external::axum::{
    self,
    extract::{Json, Query},
};
use embedder_external::serde::{Deserialize, Serialize};
use embedder_external::{fastembed, ndarray, serde_json};
use embedder_lib::{transform::CanTransform, Embedding};
use std::sync::Arc;
use tokio::time::Instant;

use crate::common::{calculate_default_batch_size, ToJsonResponse};

#[derive(Debug, Clone, Deserialize)]
pub enum OutputType {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "array")]
    Array,
    #[serde(rename = "pickle")]
    Pickle,
}

impl Default for OutputType {
    fn default() -> Self {
        Self::Json
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    #[serde(rename = "sentence-transformers/all-MiniLM-L6-v2")]
    #[cfg(feature = "sentence_transformers_all_minilm_l6_v2")]
    SentenceTransformersAllMiniLML6V2,

    #[serde(rename = "sentence-transformers/all-mpnet-base-v2")]
    #[cfg(feature = "sentence_transformers_all_mpnet_base_v2")]
    SentenceTransformerAllMpnetBaseV2,
}

macro_rules! pass_through_method {
    ($method:ident() -> $output:ty) => {
        pub fn $method(
            &self,
            documents: Vec<String>,
            batch_size: Option<usize>,
        ) -> Result<$output, EmbedderAPIError> {
            match self {
                #[cfg(feature = "sentence_transformers_all_minilm_l6_v2")]
                Self::SentenceTransformersAllMiniLML6V2 => {
                    embedder_lib::transform::models::all_minilm_l6_v2::Model::new()
                        .and_then(|model| model.$method(documents, batch_size))
                }

                #[cfg(feature = "sentence_transformers_all_mpnet_base_v2")]
                Self::SentenceTransformerAllMpnetBaseV2 => {
                    embedder_lib::transform::models::all_mpnet_base_v2::Model::new()
                        .and_then(|model| model.$method(documents, batch_size))
                }
            }
            .map_err(EmbedderAPIError::EmbedderError)
        }
    };
}

impl EmbeddingModel {
    pass_through_method!(embed_to_array() -> ndarray::Array2<f32>);
    pass_through_method!(embed_to_vec() -> Vec<Embedding>);
}

#[derive(Debug, Deserialize)]
pub struct EmbedQuery {
    output: OutputType,
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    model: EmbeddingModel,
    #[serde(default)]
    batch_size: Option<usize>,
    documents: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse<T: Serialize> {
    model: EmbeddingModel,
    duration: f32,
    embeddings: T,
}

/// The main `embed` endpoint, converting `documents` into `embeddings`.
pub async fn embed(
    Query(query): Query<EmbedQuery>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<serde_json::Value>, EmbedderAPIError> {
    let start = Instant::now();

    // Clone the model for logging only
    let model = request.model.clone();
    macro_rules! map_output_type_to_method {
        ($(OutputType::$variant:ident => $method:ident),*$(,)?) => {
            match query.output {
                $(
                    OutputType::$variant => {
                        let batch_size = request.batch_size.unwrap_or_else(|| calculate_default_batch_size(request.documents.len()));
                        let embeddings = tokio::task::spawn_blocking(move || {
                            eprintln!(
                                "Embedding {} documents to {} with batch size of {}...",
                                request.documents.len(),
                                stringify!($variant),
                                batch_size,
                            );
                            request.model.$method(request.documents, Some(batch_size))
                        })
                        .await
                        .map_err(|err| EmbedderAPIError::ConcurrencyError(err.to_string()))??;
                        EmbedResponse {
                            model,
                            duration: start.elapsed().as_secs_f32(),
                            embeddings, // Use the default `ndarray` serialization
                        }.to_json_response()
                    }
                ),*
                OutputType::Pickle => Err(EmbedderAPIError::NotImplemented("Pickle output".to_owned())),
            }
        };
    }

    map_output_type_to_method!(
        OutputType::Json => embed_to_vec,
        OutputType::Array => embed_to_array,
    )
}
