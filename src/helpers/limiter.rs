use embedder_err::EmbedderAPIError;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A simple concurrency limiter that can be used to limit the number of concurrent requests.
pub struct ConcurrencyLimiter<const L: usize> {
    counter: AtomicUsize,
}

impl<const L: usize> ConcurrencyLimiter<L> {
    /// Create a new concurrency limiter.
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }

    /// Get the current number of concurrent requests.
    pub fn current(&self) -> usize {
        self.counter.load(Ordering::SeqCst)
    }

    /// Attempt to acquire a concurrency token.
    pub fn acquire(&self) -> Result<ConcurrencyToken<'_, L>, EmbedderAPIError> {
        let previous_count = self.counter.fetch_add(1, Ordering::SeqCst);

        if previous_count >= L {
            self.release();
            eprintln!(
                "Too many concurrent requests ({}), rejecting request.",
                previous_count
            );
            Err(EmbedderAPIError::TooManyConcurrentRequests(previous_count))
        } else {
            Ok(ConcurrencyToken::new(previous_count, self))
        }
    }

    /// Decrement the counter.
    fn release(&self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A token that is issued for each request that successfully acquires the concurrency lock.
pub struct ConcurrencyToken<'a, const L: usize> {
    /// The ID of the token. This value is 0-indexed.
    pub id: usize,
    limiter: &'a ConcurrencyLimiter<L>,
}

impl<'a, const L: usize> ConcurrencyToken<'a, L> {
    /// Create a new concurrency token.
    pub fn new(id: usize, limiter: &'a ConcurrencyLimiter<L>) -> Self {
        Self { id, limiter }
    }
}

impl<const L: usize> Drop for ConcurrencyToken<'_, L> {
    fn drop(&mut self) {
        eprintln!("Concurrency token dropped, releasing lock #{}.", self.id);
        self.limiter.release();
    }
}
