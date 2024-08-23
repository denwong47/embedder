use embedder_external::axum::{
    self,
    routing::{get, post},
    Router,
};
use embedder_external::clap::Parser;

mod args;
use args::CliArgs;

mod common;

mod endpoints;

#[cfg(feature = "status")]
mod status;
#[cfg(feature = "status")]
use status::Status;

use embedder_err::EmbedderAPIError;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), EmbedderAPIError> {
    // Parse our command line arguments
    let args = CliArgs::try_parse().map_err(EmbedderAPIError::ArgsError)?;

    let socket_addr = args.socket_addr();

    // build our application with a single route
    let app = Router::new()
        .route("/", get(endpoints::root))
        .route("/embed", post(endpoints::embed));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();

    #[cfg(feature = "status")]
    Status::init();

    tokio::select!(
        _ = tokio::signal::ctrl_c() => {
            Err(EmbedderAPIError::UserTerminated)
        },
        err = axum::serve(listener, app.clone()) => {
            err.map_err(EmbedderAPIError::IoError)
        }
    )
}
