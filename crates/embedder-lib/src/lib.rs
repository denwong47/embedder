pub(crate) mod common;

pub mod transform;

pub mod reexports;

pub use fastembed::Embedding;

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::transform::{self, CanTransform};

    use std::time::Instant;

    const EPS: f32 = 1e-4;

    static TEST_DOCUMENTS: &[&'static str] = &[
        "Rust is ideal for many people for a variety of reasons. Let's look at a few of the most important groups.",
        "Rust is proving to be a productive tool for collaborating among large teams of developers with varying levels of systems programming knowledge.",
        "Low-level code is prone to various subtle bugs, which in most other languages can be caught only through extensive testing and careful code review by experienced developers.",
        "In Rust, the compiler plays a gatekeeper role by refusing to compile code with these elusive bugs, including concurrency bugs.",
        "By working alongside the compiler, the team can spend their time focusing on the program's logic rather than chasing down bugs.",
        "Rust also brings contemporary developer tools to the systems programming world:",
        "Cargo, the included dependency manager and build tool, makes adding, compiling, and managing dependencies painless and consistent across the Rust ecosystem.",
        "The Rustfmt formatting tool ensures a consistent coding style across developers.",
        "The rust-analyzer powers Integrated Development Environment (IDE) integration for code completion and inline error messages.",
        "By using these and other tools in the Rust ecosystem, developers can be productive while writing systems-level code.",
    ];

    macro_rules! create_test {
        (
            name: $name: ident,
            module: $module: ident,
            dim: $dim: literal,
            expected: $expected: expr
            $(,)?
        ) => {
            #[test]
            fn $name() {
                let model = transform::models::$module::Model::new().expect(
                    "Could not load the model."
                );

                let start = Instant::now();
                let embeddings = model.embed_to_array(
                    TEST_DOCUMENTS.to_vec(),
                    None,
                ).expect(
                    "Could not perform transformation."
                );
                println!("Time taken: {:?}", start.elapsed());

                let embeddings_with_fastembed = model.embed_with_fastembed(
                    TEST_DOCUMENTS.to_vec(),
                    None,
                ).expect(
                    "Fastembed was not able to generate embeddings."
                );

                assert_eq!(
                    embeddings.shape()[0],
                    TEST_DOCUMENTS.len(),
                    "The number of documents ({doc_len}) do not match that of the embedding shape ({emb_len}).",
                    doc_len=TEST_DOCUMENTS.len(),
                    emb_len=embeddings.shape()[0],
                );
                assert_eq!(
                    embeddings.shape()[1],
                    $dim,
                    "The number of dimensions ({dim}) do not match the expected value.",
                    dim=embeddings.shape()[1],
                );

                let expected = $expected;

                embeddings.rows().into_iter().zip(embeddings_with_fastembed).enumerate()
                .for_each(
                    |(id, (embedding, fastembed))| {
                        let expected = expected.get(id).expect("Expected embedding not found.");
                        let actual = embedding.sum();

                        assert!(
                            (actual-expected).abs() <= EPS,
                            "Mismatch for document #{id}, expected {expected}, found {actual}."
                        );

                        embedding.iter().zip(fastembed.iter()).enumerate().for_each(
                            |(fid, (actual, expected))| {
                                assert!(
                                    (actual-expected).abs() <= EPS,
                                    "Mismatch for document #{id} feature #{fid}; expected {expected} from fastembed, found {actual}.",
                                )
                            }
                        );
                    }
                );
            }
        }
    }

    create_test! (
        name: test_minilm_l6_v2,
        module: all_minilm_l6_v2,
        dim: 384,
        expected: &[
            -0.189222_f32,
            -0.062057_f32,
            -0.066258_f32,
            0.026673_f32,
            0.082889_f32,
            0.044613_f32,
            0.268277_f32,
            0.389151_f32,
            0.315552_f32,
            0.084336_f32,
        ],
    );

    create_test!(
        name: test_mpnet_base_v2,
        module: all_mpnet_base_v2,
        dim: 768,
        expected: &[
            -0.211852_f32,
            -0.082862_f32,
            -0.173984_f32,
            0.018110_f32,
            0.034919_f32,
            0.023689_f32,
            -0.080518_f32,
            0.143173_f32,
            0.076229_f32,
            0.0506501_f32,
        ],
    );
}
