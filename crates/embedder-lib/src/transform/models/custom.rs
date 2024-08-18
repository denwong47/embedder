use crate::transform::traits::CanTransform;

use embedder_err::EmbedderError;
use embedder_external::fastembed;
use std::sync::Arc;

pub struct Model {
    name: &'static str,
    output_key: &'static str,
    model: fastembed::TextEmbedding,
    pooling: Option<fastembed::Pooling>,
}

impl Model {
    /// Create a new instance of the model.
    ///
    /// To create the model using a folder containing the model files, use the `from_path` method.
    pub fn new(
        name: &'static str,
        output_key: &'static str,
        onnx_file: Vec<u8>,
        tokenizer_file: Vec<u8>,
        config_file: Vec<u8>,
        special_tokens_map_file: Vec<u8>,
        tokenizer_config_file: Vec<u8>,
        pooling: Option<fastembed::Pooling>,
        quantization: fastembed::QuantizationMode,
    ) -> Result<Arc<Self>, EmbedderError> {
        let user_model = fastembed::UserDefinedEmbeddingModel {
            onnx_file,
            tokenizer_files: fastembed::TokenizerFiles {
                tokenizer_file,
                config_file,
                special_tokens_map_file,
                tokenizer_config_file,
            },
            pooling: pooling.clone(),
            quantization,
        };

        fastembed::TextEmbedding::try_new_from_user_defined(user_model, Default::default())
            .map_err(|err| EmbedderError::ModelLoadError {
                name: &name,
                error: err.to_string(),
            })
            .map(|text_embedding| {
                Arc::new(Self {
                    name,
                    output_key,
                    model: text_embedding,
                    pooling,
                })
            })
    }

    /// Create a new instance of the model from a folder containing the model files.
    ///
    /// The folder should contain the following files:
    /// - `tokenizer.json`
    /// - `config.json`
    /// - `special_tokens_map.json`
    /// - `tokenizer_config.json`
    /// - and the model file named by `model_file`, which is commonly `model.onnx`.
    ///
    /// Different from ``MODEL_PATH``, the `path` parameter should be the path
    /// to the folder containing the model files, rather than the path to the collection
    /// of models within subfolders.
    ///
    /// The returned model will be an [`Arc`] to the model.
    pub fn from_path(
        name: &'static str,
        output_key: &'static str,
        path: &std::path::Path,
        model_file: &str,
        pooling: Option<fastembed::Pooling>,
        quantization: fastembed::QuantizationMode,
    ) -> Result<Arc<Self>, EmbedderError> {
        let path = path.join(name);

        let onnx_file =
            std::fs::read(path.join(model_file)).map_err(|err| EmbedderError::ModelPathError {
                path: path.to_owned().into(),
                error: err,
            })?;
        let tokenizer_file = std::fs::read(path.join("tokenizer.json")).map_err(|err| {
            EmbedderError::ModelPathError {
                path: path.to_owned().into(),
                error: err,
            }
        })?;
        let config_file = std::fs::read(path.join("config.json")).map_err(|err| {
            EmbedderError::ModelPathError {
                path: path.to_owned().into(),
                error: err,
            }
        })?;
        let special_tokens_map_file =
            std::fs::read(path.join("special_tokens_map.json")).map_err(|err| {
                EmbedderError::ModelPathError {
                    path: path.to_owned().into(),
                    error: err,
                }
            })?;
        let tokenizer_config_file =
            std::fs::read(path.join("tokenizer_config.json")).map_err(|err| {
                EmbedderError::ModelPathError {
                    path: path.to_owned().into(),
                    error: err,
                }
            })?;

        Self::new(
            name,
            output_key,
            onnx_file,
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
            pooling,
            quantization,
        )
    }
}

impl CanTransform for Arc<Model> {
    /// The name of the model.
    fn name(&self) -> &str {
        &self.name
    }

    /// The output key of the model.
    fn output_key(&self) -> &'static str {
        self.output_key
    }

    /// The pooling method used by the model.
    fn pooling(&self) -> Option<fastembed::Pooling> {
        self.pooling.clone()
    }

    fn transform<'e, 'r, 's, S: AsRef<str> + Send + Sync>(
        &'e self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<fastembed::EmbeddingOutput<'r, 's>, EmbedderError>
    where
        'e: 'r,
        'e: 's,
    {
        self.model
            .transform(texts, batch_size)
            .map_err(|err| EmbedderError::FastEmbedError(err))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils;

    #[cfg(feature = "sentence_transformers_all_minilm_l6_v2")]
    #[test]
    fn load_all_mini_l6_v2_as_custom() {
        let model_path_str = test_utils::get_model_path();
        let model_path = std::path::Path::new(&model_path_str)
            .canonicalize()
            .expect(&format!(
                "Failed to get the canonical path for {model_path:?}.",
                model_path = &model_path_str,
            ));

        let model = super::Model::from_path(
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence_embedding",
            &model_path,
            "model.onnx",
            Some(fastembed::Pooling::Mean),
            fastembed::QuantizationMode::None,
        )
        .expect(&format!(
            "Failed to load the model. Make sure the model files are present in {model_path:?}.",
            model_path = &model_path,
        ));

        let documents = vec!["Hello, world!", "This is a test."];

        let embeddings = model
            .embed_to_array(documents.clone(), None)
            .expect("Failed to embed the documents.");

        assert_eq!(embeddings.shape()[0], documents.len());
        assert_eq!(embeddings.shape()[1], 384);

        let expected = &[0.5960535, 0.28592288];

        embeddings
            .rows()
            .into_iter()
            .zip(expected.iter())
            .enumerate()
            .for_each(|(id, (embedding, expected))| {
                let actual = embedding.sum();

                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "Mismatch for document #{id}, expected {expected}, found {actual}."
                );
            });
    }
}
