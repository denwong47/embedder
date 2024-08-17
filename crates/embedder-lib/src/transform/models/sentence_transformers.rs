//! Models from [Sentence Transformers].
//!
//! [Sentence Transformers]: https://huggingface.co/sentence-transformers
//!

use super::binaries;
use crate::transform::traits::CanTransform;

use embedder_err::EmbedderError;
use std::sync::{Arc, OnceLock};

macro_rules! create_model {
    (
        module: $module: ident,
        name: $name: literal,
        feature: $feature: literal,
        binaries: $binaries: ident,
        output_key: $output_key: literal,
        pooling: $pooling: expr,
        quantization: $quantization: expr
        $(,)?
    ) => {
        #[cfg(feature=$feature)]
        pub mod $module {
            use super::*;

            static MODEL: OnceLock<Result<Arc<Model>, EmbedderError>> = OnceLock::new();

            /// Model for the `all-mpnet-base-v2` model.
            ///
            /// This has private fields, preventing instantiation
            /// without the `new` method.
            pub struct Model {
                model: fastembed::TextEmbedding,
            }

            impl Model {
                const NAME: &'static str = $name;

                #[cfg(test)]
                pub fn embed_with_fastembed<S: AsRef<str> + Send + Sync>(
                    &self,
                    texts: Vec<S>,
                    batch_size: Option<usize>,
                ) -> Result<Vec<fastembed::Embedding>, EmbedderError> {
                    self.model
                        .embed(texts, batch_size)
                        .map_err(|err| EmbedderError::FastEmbedError(err))
                }

                /// Create a new instance of the model.
                pub fn new() -> Result<Arc<Self>, EmbedderError> {
                    match MODEL.get_or_init(|| {
                        let user_model = fastembed::UserDefinedEmbeddingModel {
                            onnx_file: binaries::$binaries::MODEL_FILE.to_vec(),
                            tokenizer_files: fastembed::TokenizerFiles {
                                tokenizer_file: binaries::$binaries::TOKENIZER_FILE.to_vec(),
                                config_file: binaries::$binaries::CONFIG_FILE.to_vec(),
                                special_tokens_map_file:
                                    binaries::$binaries::SPECIAL_TOKENS_MAP_FILE.to_vec(),
                                tokenizer_config_file: binaries::$binaries::TOKENIZER_CONFIG_FILE
                                    .to_vec(),
                            },
                            pooling: Self::POOLING,
                            quantization: $quantization,
                        };

                        fastembed::TextEmbedding::try_new_from_user_defined(
                            user_model,
                            Default::default(),
                        )
                        .map_err(|err| EmbedderError::FastEmbedError(err))
                        .map(|text_embedding| {
                            Arc::new(Self {
                                model: text_embedding,
                            })
                        })
                    }) {
                        &Ok(ref model) => Ok(Arc::clone(model)),
                        &Err(ref err) => Err(EmbedderError::ModelLoadError {
                            name: Self::NAME,
                            error: err.to_string(),
                        }),
                    }
                }
            }

            impl CanTransform for Model {
                const OUTPUT_KEY: &'static str = $output_key;
                const POOLING: Option<fastembed::Pooling> = $pooling;

                /// Transforms the input texts into embeddings.
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
        }
    };
}

create_model!(
    module: all_mpnet_base_v2,
    name: "sentence-transformers/all-mpnet-base-v2",
    feature: "sentence_transformers_all_mpnet_base_v2",
    binaries: sentence_transformers_all_mpnet_base_v2,
    output_key: "sentence_embedding",
    pooling: Some(fastembed::Pooling::Mean),
    quantization: fastembed::QuantizationMode::None,
);
create_model!(
    module: all_minilm_l6_v2,
    name: "sentence-transformers/all-mpnet-base-v2",
    feature: "sentence_transformers_all_minilm_l6_v2",
    binaries: sentence_transformers_all_minilm_l6_v2,
    output_key: "sentence_embedding",
    pooling: Some(fastembed::Pooling::Mean),
    quantization: fastembed::QuantizationMode::None,
);
