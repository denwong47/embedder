use std::env;

use embedder_err::EmbedderError;

pub const DEFAULT_MODEL_PATH: &str = "./models";

/// Get an environment variable and parse it into the desired type;
/// if the variable is not set or invalid, return the default value.
pub fn get_env_var<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    env::var(key)
        .map_err(|error| EmbedderError::EnvVarError {
            key: key.to_string(),
            error: error.to_string(),
        })
        .and_then(|value| {
            if value.len() == 0 {
                Err(EmbedderError::EnvVarError {
                    key: key.to_string(),
                    error: "Empty value".to_string(),
                })
            } else {
                Ok(value)
            }
        })
        .and_then(|value| {
            value
                .parse()
                .map_err(
                    |error: <T as std::str::FromStr>::Err| EmbedderError::EnvVarError {
                        key: key.to_string(),
                        error: format!("{:?}", error),
                    },
                )
        })
        .unwrap_or(default)
}

/// Get the path to the model from the environment variable `MODEL_PATH`.
pub fn get_model_path() -> String {
    get_env_var("MODEL_PATH", DEFAULT_MODEL_PATH.to_owned())
}
