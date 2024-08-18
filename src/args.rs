use embedder_external::clap::{self, Parser};

use std::net::Ipv4Addr;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    /// The host IP of the server. Defaults to all interfaces.
    #[arg(long, default_value_t = Ipv4Addr::new(0, 0, 0, 0))]
    host: Ipv4Addr,

    /// The port to listen on. Defaults to 3000.
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
}

impl CliArgs {
    /// Get the socket address from the host and port.
    pub fn socket_addr(&self) -> std::net::SocketAddr {
        std::net::SocketAddr::new(self.host.into(), self.port)
    }
}
