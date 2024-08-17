use std::{env, fs, io, path};

use unicode_ident;

/// Build module name
fn build_module_name(name: &str) -> Option<String> {
    name.chars()
        .fold((None, true), |(acc, mut word_start), c| {
            match (acc, c) {
                (None, c) if unicode_ident::is_xid_start(c) => {
                    // Empty string,
                    (Some(c.to_lowercase().to_string()), false)
                }
                (Some(mut acc), c) if unicode_ident::is_xid_continue(c) && c != '_' => {
                    // Existing name appended with Valid character
                    word_start = false;
                    acc.push(c.to_ascii_lowercase());
                    (Some(acc), word_start)
                }
                (Some(mut acc), _) => {
                    if !word_start {
                        // End of word, add underscore
                        acc.push('_');
                    }
                    // Invalid character, treat as space
                    (Some(acc), true)
                }
                (acc, _) => {
                    // Invalid beginning character, ignore
                    (acc, true)
                }
            }
        })
        .0
}

/// Add a model to the binaries library
fn add_model(model_path: &str, name: &str, file_names: [&str; 5]) -> io::Result<String> {
    let model_dir = path::Path::new(model_path).join(name);
    println!("cargo::rerun-if-changed={path}", path = model_dir.display());

    let module_codes = Result::<Vec<String>, io::Error>::from_iter(
        file_names
            .iter()
            .zip([
                "MODEL_FILE",
                "TOKENIZER_FILE",
                "CONFIG_FILE",
                "SPECIAL_TOKENS_MAP_FILE",
                "TOKENIZER_CONFIG_FILE",
            ])
            .map(|(file_name, var_name)| {
                let file_path = model_dir.join(file_name);

                Ok(format!(
                    r#"    pub const {var_name}: &[u8] = include_bytes!({file_path:?});"#,
                    var_name = var_name,
                    file_path = file_path
                ))
            }),
    )?;

    Ok(format!(
        "/// Binaries for '{name}'.\n\
        pub mod {module_name} {{\n\
            {module_codes}\n\
        }}\
        ",
        name = name,
        module_name = build_module_name(name).ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Model {name} cannot be converted to a UAX#31 compatible identifier.",
                name = name
            )
        ))?,
        module_codes = module_codes.join("\n\n")
    ))
}

fn main() -> io::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    // Default to the `models` directory if the `MODEL_PATH` environment variable is not set.
    let model_dir = env::var("MODEL_PATH").unwrap_or_else(|_| "./models".to_string());

    let dest_path = path::Path::new(&out_dir).join("src/transform/models/binaries.rs");

    dest_path.parent().map(|parent| fs::create_dir_all(parent));

    let models = [
        #[cfg(feature = "sentence_transformers_all_minilm_l6_v2")]
        add_model(
            &model_dir,
            "sentence-transformers/all-MiniLM-L6-v2",
            [
                "model.onnx",
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
        )?,
        #[cfg(feature = "sentence_transformers_all_mpnet_base_v2")]
        add_model(
            &model_dir,
            "sentence-transformers/all-mpnet-base-v2",
            [
                "model.onnx",
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
        )?,
    ];

    fs::write(&dest_path, models.join("\n\n"))?;

    // Tell Cargo that if the english words changes, to rerun this build script.
    // println!(format!"cargo::rerun-if-changed=data/words/en_common_words.csv");

    Ok(())
}
