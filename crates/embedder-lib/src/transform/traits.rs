//! Traits for embedding models.
//!

use crate::Embedding;
use embedder_err::EmbedderError;

const EPS: f32 = 1e-12;

pub trait CanTransform {
    /// The chosen key for the output embeddings.
    const OUTPUT_KEY: &'static str = "sentence_embedding";
    const POOLING: Option<fastembed::Pooling> = None;

    /// The key for the output embeddings.
    ///
    /// This should only provide one key - test the model to see what keys it produces,
    /// and choose the one that produce the desired embeddings.
    ///
    /// Instead of trying different keys in succession, the embedding operation
    /// should fail if the key is not found.
    fn output_precedence() -> &'static [fastembed::OutputKey] {
        &[fastembed::OutputKey::ByName(Self::OUTPUT_KEY)]
    }

    /// Static function to converts the output to a 2D array.
    fn output_to_2d_array<'r, 's>(
        output: fastembed::EmbeddingOutput<'r, 's>,
    ) -> Result<ndarray::Array2<f32>, EmbedderError> {
        output
            .export_with_transformer(
                // This needs to be `anyhow::Result`
                |batches| {
                    batches
                        .iter()
                        .map(|batch| {
                            batch.select_and_pool_output(&Self::output_precedence(), Self::POOLING)
                        })
                        .reduce(|acc, res| match (acc, res) {
                            (Err(e), _) => return Err(e),
                            (_, Err(e)) => return Err(e),
                            (Ok(acc), Ok(res)) => {
                                ndarray::concatenate(ndarray::Axis(0), &[acc.view(), res.view()])
                                    .map_err(|err| anyhow::Error::from(err))
                            }
                        })
                        .unwrap_or(Err(anyhow::Error::msg("No output found.")))
                },
            )
            .map_err(EmbedderError::FastEmbedError)
            .map(
                |mut array: ndarray::ArrayBase<
                    ndarray::OwnedRepr<f32>,
                    ndarray::Dim<[usize; 2]>,
                >| {
                    // Normalize the embeddings
                    array.rows_mut().into_iter().for_each(|mut row| {
                        let norm = row.map(|v| v.powi(2)).sum().sqrt();
                        for val in row.iter_mut() {
                            *val /= norm + EPS;
                        }
                    });

                    array
                },
            )
    }

    /// Transforms the input texts into embeddings.
    fn transform<'e, 'r, 's, S: AsRef<str> + Send + Sync>(
        &'e self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<fastembed::EmbeddingOutput<'r, 's>, EmbedderError>
    where
        'e: 'r,
        'e: 's;

    /// Exports the model to a [`ndarray::Array2<f32>`].
    fn embed_to_array<'e, 'r, 's, S: AsRef<str> + Send + Sync>(
        &'e self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<ndarray::Array2<f32>, EmbedderError>
    where
        'e: 'r,
        'e: 's,
    {
        let output = self.transform(texts, batch_size)?;
        Self::output_to_2d_array(output)
    }

    /// Exports the model to a [`Vec<Embedding>`].
    fn embed_to_vec<'e, 'r, 's, S: AsRef<str> + Send + Sync>(
        &'e self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>, EmbedderError>
    where
        'e: 'r,
        'e: 's,
    {
        Ok(self
            .embed_to_array(texts, batch_size)?
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>())
    }
}
