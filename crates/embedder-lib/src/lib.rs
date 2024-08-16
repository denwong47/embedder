#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use std::path::PathBuf;

    use fastembed::{
        EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding,
        TokenizerFiles, UserDefinedEmbeddingModel,
    };

    use super::*;

    #[test]
    fn embed_simple() {
        macro_rules! local_model {
            ($folder:literal) => {
                UserDefinedEmbeddingModel {
                    onnx_file: include_bytes!(concat!($folder, "/model.onnx")).to_vec(),
                    tokenizer_files: TokenizerFiles {
                        tokenizer_file: include_bytes!(concat!($folder, "/tokenizer.json"))
                            .to_vec(),
                        config_file: include_bytes!(concat!($folder, "/config.json")).to_vec(),
                        special_tokens_map_file: include_bytes!(concat!(
                            $folder,
                            "/special_tokens_map.json"
                        ))
                        .to_vec(),
                        tokenizer_config_file: include_bytes!(concat!(
                            $folder,
                            "/tokenizer_config.json"
                        ))
                        .to_vec(),
                    },
                    pooling: Some(Pooling::Cls),
                }
            };
        }

        // With custom InitOptions
        let model: TextEmbedding = TextEmbedding::try_new_from_user_defined(
            local_model!("/Users/denwong47/Documents/repos/embedder/models/all-MiniLM-L6-v2"),
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
