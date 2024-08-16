#[cfg(test)]
pub(crate) mod test_utils;

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use embedder_err::EmbedderError;
    use fastembed::{
        EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, QuantizationMode,
        TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
    };
    use std::path::{Path, PathBuf};

    use super::*;

    /// Load bytes from a file, with a nicer error message.
    fn load_bytes_from_file(path: &Path) -> Result<Vec<u8>, EmbedderError> {
        std::fs::read(path).map_err(|error| EmbedderError::ModelPathError {
            path: path.into(),
            error,
        })
    }

    #[test]
    fn embed_simple() {
        let model_path = PathBuf::from(test_utils::get_env_var(
            "MODEL_PATH",
            test_utils::DEFAULT_MODEL_PATH.to_owned(),
        ))
        .canonicalize()
        .unwrap();

        fn local_model(path: &std::path::Path) -> Result<UserDefinedEmbeddingModel, EmbedderError> {
            Ok(UserDefinedEmbeddingModel {
                onnx_file: load_bytes_from_file(&path.join("model.onnx"))?.to_vec(),
                tokenizer_files: TokenizerFiles {
                    tokenizer_file: load_bytes_from_file(&path.join("tokenizer.json"))?.to_vec(),
                    config_file: load_bytes_from_file(&path.join("config.json"))?.to_vec(),
                    special_tokens_map_file: load_bytes_from_file(
                        &path.join("special_tokens_map.json"),
                    )?
                    .to_vec(),
                    tokenizer_config_file: load_bytes_from_file(
                        &path.join("tokenizer_config.json"),
                    )?
                    .to_vec(),
                },
                pooling: Some(Pooling::Mean),
                quantization: QuantizationMode::None,
            })
        }

        let model_folder = model_path.join("sentence-transformers/all-MiniLM-L6-v2");
        // With custom InitOptions
        let model: TextEmbedding = TextEmbedding::try_new_from_user_defined(
            local_model(&model_folder).unwrap(),
            InitOptionsUserDefined::default(),
        )
        .expect("Failed to load model");

        let documents = vec![
            "This is an example sentence",
            "Each sentence is converted",
            "Yet another sentence",
            "The last sentence",
        ];

        // Generate embeddings with the default batch size, 256
        let embeddings = model
            .embed(documents, None)
            .expect("Failed to generate embeddings");

        dbg!(embeddings.len()); // -> Embeddings length: 2
        dbg!(embeddings[0].len()); // -> Embedding dimension: 384

        // dbg!(embeddings);
    }
}
