//! Module to hold the status of the server.
//!
//! This is implemented globally as a singleton, so the status would not be accurate if
//! multiple instances of the server were running - which is not a supported use case.
use embedder_external::serde::{self, ser::SerializeStruct, Serialize};
use memory_stats::memory_stats;
use tokio::time::Instant;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// The global status of the server.
static GLOBAL_STATUS: OnceLock<Arc<Status>> = OnceLock::new();

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryUsage {
    physical_used: usize,
    virtual_used: usize,
}

impl MemoryUsage {
    pub fn new() -> Option<Self> {
        let memory_stats::MemoryStats {
            physical_mem,
            virtual_mem,
        } = memory_stats()?;

        Some(Self {
            physical_used: physical_mem,
            virtual_used: virtual_mem,
        })
    }
}

/// A singleton struct to hold the status of the server.
#[derive(Debug)]
pub struct Status {
    start_time: Instant,
    requests: AtomicUsize,
}

impl Serialize for Status {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Status", 3)?;
        state.serialize_field("uptime", &self.start_time.elapsed().as_secs_f32())?;
        state.serialize_field("requests", &self.requests.load(Ordering::Relaxed))?;
        state.serialize_field("memory", &MemoryUsage::new())?;
        state.end()
    }
}

impl Status {
    /// Initialize the status, without returning it.
    pub fn init() {
        Self::get();
    }

    /// Create a new instance of the status.
    pub fn get() -> Arc<Self> {
        Arc::clone(GLOBAL_STATUS.get_or_init(|| {
            Arc::new(Self {
                start_time: Instant::now(),
                requests: AtomicUsize::new(0),
            })
        }))
    }

    /// Increment the number of requests.
    #[allow(dead_code)]
    pub fn increment_requests(&self) {
        self.requests.fetch_add(1, Ordering::Relaxed);
    }
}
