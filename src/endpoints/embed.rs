//! The main `embed` endpoint, converting `documents` into `embeddings`.

use embedder_err::EmbedderAPIError;
use embedder_external::axum::{
    self,
    extract::{Json, Query},
};
use embedder_external::serde::{Deserialize, Serialize};
use embedder_lib::{transform::CanTransform, Embedding};
use std::sync::Arc;
use tokio::time::Instant;

#[derive(Debug, Clone, Deserialize)]
pub enum OutputType {
    #[serde(rename = "json")]
    Json,
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

impl EmbeddingModel {
    pub fn embed_to_vec(&self, documents: Vec<String>) -> Result<Vec<Embedding>, EmbedderAPIError> {
        match self {
            #[cfg(feature = "sentence_transformers_all_minilm_l6_v2")]
            Self::SentenceTransformersAllMiniLML6V2 => {
                embedder_lib::transform::models::all_minilm_l6_v2::Model::new()
                    .and_then(|model| model.embed_to_vec(documents, Some(32)))
            }

            #[cfg(feature = "sentence_transformers_all_mpnet_base_v2")]
            Self::SentenceTransformerAllMpnetBaseV2 => {
                embedder_lib::transform::models::all_mpnet_base_v2::Model::new()
                    .and_then(|model| model.embed_to_vec(documents, Some(32)))
            }
        }
        .map_err(EmbedderAPIError::EmbedderError)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbedQuery {
    output: OutputType,
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    model: EmbeddingModel,
    documents: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    model: EmbeddingModel,
    duration: f32,
    embeddings: Vec<Vec<f32>>,
}

/// The main `embed` endpoint, converting `documents` into `embeddings`.
pub async fn embed(
    Query(query): Query<EmbedQuery>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, EmbedderAPIError> {
    let start = Instant::now();

    match query.output {
        OutputType::Json => {
            let model = request.model.clone();
            let embeddings = tokio::task::spawn_blocking(move || {
                eprintln!("Embedding {} documents...", request.documents.len());
                request.model.embed_to_vec(request.documents)
            })
            .await
            .map_err(|err| EmbedderAPIError::ConcurrencyError(err.to_string()))??;
            Ok(Json(EmbedResponse {
                model,
                duration: start.elapsed().as_secs_f32(),
                embeddings,
            }))
        }
        OutputType::Pickle => Err(EmbedderAPIError::NotImplemented("Pickle output".to_owned())),
    }
}
