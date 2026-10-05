use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::projection::ProjectionFile;
use super::resolve::{resolve_words, Resolution};
use super::tables::{load_stopwords, load_vocabulary, Vocabulary};
use super::vector::{sparse_vector, MinicoilConstants, Role};
use super::{MinicoilError, MinicoilTables, ProjectionRows};

const FIXTURES: [&str; 8] = [
    "machine-learning-document",
    "machine-learning-question",
    "bear-document",
    "bear-question",
    "cafe-document",
    "cafe-question",
    "tessera-document",
    "tessera-question",
];
const VALUE_TOLERANCE: f32 = 0.000_01;

#[derive(Deserialize)]
struct Fixture {
    role: String,
    tokens: Vec<String>,
    token_vectors: Vec<Vec<f32>>,
    words: Vec<FixtureWord>,
    projection_rows: HashMap<String, Vec<Vec<f32>>>,
    constants: FixtureConstants,
    sparse: FixtureSparse,
}

#[derive(Deserialize)]
struct FixtureWord {
    word: String,
    token_positions: Vec<usize>,
    resolution: String,
    vocab_id: u32,
}

#[derive(Deserialize)]
struct FixtureConstants {
    k: f64,
    b: f64,
    avg_len: f64,
    gap: u32,
    vocab_size: u32,
    token_max_length: usize,
}

#[derive(Deserialize)]
struct FixtureSparse {
    indices: Vec<u32>,
    values: Vec<f32>,
}

/// The excerpt of the real vocabulary, stem mapping and stop words that the
/// fixture texts touch (see `testdata/tables.json`'s `source`).
#[derive(Deserialize)]
struct TablesExcerpt {
    vocab_size: u32,
    words: HashMap<String, String>,
    stem_mapping: HashMap<String, String>,
    stopwords: Vec<String>,
}

fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> Fixture {
    let path = repository().join(format!("certification/fixtures/minicoil/{name}.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn tables() -> MinicoilTables {
    let path = repository().join("src/encoding/minicoil/testdata/tables.json");
    let excerpt: TablesExcerpt =
        serde_json::from_slice(&std::fs::read(path).expect("tables excerpt")).expect("excerpt");
    let vocabulary = Vocabulary {
        words: excerpt
            .words
            .into_iter()
            .map(|(id, word)| (id.parse().expect("numeric id"), word))
            .collect(),
        stem_mapping: excerpt.stem_mapping,
        vocab_size: excerpt.vocab_size,
    };
    MinicoilTables::new(vocabulary, excerpt.stopwords.into_iter().collect())
}

fn resolution_name(resolution: Resolution) -> &'static str {
    match resolution {
        Resolution::StopWord => "stop_word",
        Resolution::Exact => "exact",
        Resolution::StemMapping => "stem_mapping",
        Resolution::Stemmed => "stemmed",
        Resolution::Unknown => "unknown",
    }
}

#[test]
fn word_resolution_matches_every_fixture() {
    let tables = tables();
    for name in FIXTURES {
        let fixture = fixture(name);
        let tokens = fixture
            .tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let resolved = resolve_words(&tokens, &tables);
        let actual = resolved
            .iter()
            .map(|word| {
                (
                    word.word.as_str(),
                    word.token_positions.clone(),
                    resolution_name(word.resolution),
                    word.vocab_id,
                )
            })
            .collect::<Vec<_>>();
        let expected = fixture
            .words
            .iter()
            .map(|word| {
                (
                    word.word.as_str(),
                    word.token_positions.clone(),
                    word.resolution.as_str(),
                    word.vocab_id,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn sparse_vectors_match_every_fixture() {
    let tables = tables();
    for name in FIXTURES {
        let fixture = fixture(name);
        let tokens = fixture
            .tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let vectors = fixture.token_vectors.concat();
        let mut rows = ProjectionRows::default();
        for (id, matrix) in &fixture.projection_rows {
            rows.insert(id.parse().expect("numeric id"), matrix.concat());
        }
        let role = match fixture.role.as_str() {
            "document" => Role::Document,
            "question" => Role::Question,
            other => panic!("{name}: role {other}"),
        };
        let constants = MinicoilConstants {
            k: fixture.constants.k,
            b: fixture.constants.b,
            avg_len: fixture.constants.avg_len,
            gap: fixture.constants.gap,
            token_max_length: fixture.constants.token_max_length,
        };
        assert_eq!(constants, MinicoilConstants::default(), "{name} constants");
        assert_eq!(tables.vocab_size(), fixture.constants.vocab_size, "{name}");

        let sparse = sparse_vector(&tokens, &vectors, role, &tables, &rows, &constants)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(sparse.indices, fixture.sparse.indices, "{name} indices");
        let largest = sparse
            .values
            .iter()
            .zip(&fixture.sparse.values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        println!("{name}: indices equal, largest absolute value difference {largest:e}");
        assert!(largest <= VALUE_TOLERANCE, "{name}: {largest}");
    }
}

#[test]
fn mismatched_token_vectors_are_refused() {
    let error = sparse_vector(
        &["[CLS]", "[SEP]"],
        &[0.0; 512],
        Role::Document,
        &tables(),
        &ProjectionRows::default(),
        &MinicoilConstants::default(),
    )
    .expect_err("one vector for two tokens");
    assert!(matches!(
        error,
        MinicoilError::TokenVectors {
            tokens: 2,
            values: 512,
            expected: 1024
        }
    ));
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let directory =
            std::env::temp_dir().join(format!("tessera-minicoil-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        Self(directory)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).expect("scratch file");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A version-1 `.npy` file with the given header dictionary and data.
fn npy(header: &str, data: &[u8]) -> Vec<u8> {
    let mut dictionary = header.to_string();
    // NumPy pads the header with spaces and ends it with a newline so the data starts on a
    // 64-byte boundary.
    while !(10 + dictionary.len() + 1).is_multiple_of(64) {
        dictionary.push(' ');
    }
    dictionary.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(
        &u16::try_from(dictionary.len())
            .expect("short header")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(dictionary.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn projection_data(rows: u32) -> Vec<u8> {
    (0..rows * 2048)
        .flat_map(|index| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small test indices are exact in float32"
            )]
            let value = index as f32;
            value.to_le_bytes()
        })
        .collect()
}

const GOOD_HEADER: &str = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 512, 4), }";

#[test]
fn projection_file_reads_requested_rows() {
    let scratch = Scratch::new("projection-ok");
    let path = scratch.write("weights.npy", &npy(GOOD_HEADER, &projection_data(3)));
    let mut file = ProjectionFile::open(&path).expect("valid projection");
    assert_eq!(file.vocab_rows(), 3);
    let rows = file.read_rows(&[2, 1, 2]).expect("rows");
    let row = rows.get(2).expect("row 2");
    assert_eq!(row.len(), 2048);
    assert!((row[0] - 4096.0).abs() < f32::EPSILON);
    assert!((row[2047] - 6143.0).abs() < f32::EPSILON);
    assert!(rows.get(1).is_some());
    assert!(rows.get(0).is_none());
    assert!(matches!(
        file.read_rows(&[3]),
        Err(MinicoilError::MissingRow { vocab_id: 3 })
    ));
}

#[test]
fn projection_file_refuses_each_bad_header() {
    let scratch = Scratch::new("projection-bad");
    let data = projection_data(3);
    let cases: [(&str, Vec<u8>); 7] = [
        ("magic", b"NOTNUMPY-file".to_vec()),
        (
            "type",
            npy(
                "{'descr': '>f4', 'fortran_order': False, 'shape': (3, 512, 4), }",
                &data,
            ),
        ),
        (
            "wide",
            npy(
                "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 512, 4), }",
                &data,
            ),
        ),
        (
            "order",
            npy(
                "{'descr': '<f4', 'fortran_order': True, 'shape': (3, 512, 4), }",
                &data,
            ),
        ),
        (
            "rank",
            npy(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (1536, 4), }",
                &data,
            ),
        ),
        (
            "width",
            npy(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 256, 8), }",
                &data,
            ),
        ),
        ("length", npy(GOOD_HEADER, &data[..data.len() - 4])),
    ];
    for (name, bytes) in cases {
        let path = scratch.write(&format!("{name}.npy"), &bytes);
        let error = ProjectionFile::open(&path).expect_err(name);
        let named = match name {
            "magic" => matches!(error, MinicoilError::ProjectionMagic { .. }),
            "type" | "wide" => matches!(error, MinicoilError::ProjectionType { .. }),
            "order" => matches!(error, MinicoilError::ProjectionOrder { .. }),
            "rank" | "width" => matches!(error, MinicoilError::ProjectionShape { .. }),
            _ => matches!(
                error,
                MinicoilError::ProjectionLength {
                    expected: 24_576,
                    actual: 24_572,
                    ..
                }
            ),
        };
        assert!(named, "{name}: {error}");
    }
}

#[test]
fn vocabulary_loader_numbers_words_from_one() {
    let scratch = Scratch::new("vocabulary");
    let path = scratch.write(
        "vocab.json",
        br#"{"vocab": ["bear", "machin"], "stem_mapping": {"bear": "bear", "machin": "machin"}}"#,
    );
    let vocabulary = load_vocabulary(&path).expect("vocabulary");
    assert_eq!(vocabulary.vocab_size, 3);
    assert_eq!(vocabulary.words.get(&1).map(String::as_str), Some("bear"));
    assert_eq!(vocabulary.words.get(&2).map(String::as_str), Some("machin"));
    assert_eq!(vocabulary.stem_mapping.len(), 2);

    let dangling = scratch.write(
        "dangling.json",
        br#"{"vocab": ["bear"], "stem_mapping": {"tree": "tree"}}"#,
    );
    assert!(matches!(
        load_vocabulary(&dangling),
        Err(MinicoilError::Vocabulary { .. })
    ));
    let malformed = scratch.write("malformed.json", br#"{"vocab": ["bear"]}"#);
    assert!(matches!(
        load_vocabulary(&malformed),
        Err(MinicoilError::Vocabulary { .. })
    ));
    assert!(matches!(
        load_vocabulary(&scratch.0.join("absent.json")),
        Err(MinicoilError::Io { .. })
    ));
}

#[test]
fn stopword_loader_reads_one_word_per_line() {
    let scratch = Scratch::new("stopwords");
    let path = scratch.write("stopwords.txt", b"the\na\r\nis\n");
    let stopwords = load_stopwords(&path).expect("stop words");
    let expected = ["the", "a", "is"]
        .into_iter()
        .map(str::to_string)
        .collect::<HashSet<_>>();
    assert_eq!(stopwords, expected);
}

#[test]
fn every_resolution_branch_in_order() {
    // Made-up tables: "bear" is a word, "bore" maps to it directly, "bears" reaches it by
    // its stem, and "the" is a stop word even though the stem mapping also knows it.
    let vocabulary = Vocabulary {
        words: HashMap::from([(1, "bear".to_string())]),
        stem_mapping: HashMap::from([
            ("bore".to_string(), "bear".to_string()),
            ("bear".to_string(), "bear".to_string()),
            ("the".to_string(), "bear".to_string()),
        ]),
        vocab_size: 2,
    };
    let tables = MinicoilTables::new(vocabulary, HashSet::from(["the".to_string()]));
    let tokens = ["the", "bear", "bore", "bears", "ca", "##ve"];
    let resolved = resolve_words(&tokens, &tables)
        .into_iter()
        .map(|word| (word.word, word.resolution, word.vocab_id))
        .collect::<Vec<_>>();
    assert_eq!(
        resolved,
        [
            ("the".to_string(), Resolution::StopWord, 0),
            ("bear".to_string(), Resolution::Exact, 1),
            ("bore".to_string(), Resolution::StemMapping, 1),
            ("bears".to_string(), Resolution::Stemmed, 1),
            ("cave".to_string(), Resolution::Unknown, 0),
        ]
    );
}

#[test]
fn token_output_glue_matches_direct_conversion_for_every_fixture() {
    let tables = tables();
    for name in FIXTURES {
        let fixture = fixture(name);
        let vectors = fixture.token_vectors.concat();
        let mut rows = ProjectionRows::default();
        for (id, matrix) in &fixture.projection_rows {
            rows.insert(id.parse().expect("numeric id"), matrix.concat());
        }
        let role = match fixture.role.as_str() {
            "document" => Role::Document,
            "question" => Role::Question,
            other => panic!("unexpected role {other}"),
        };
        let borrowed = fixture
            .tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let expected = sparse_vector(
            &borrowed,
            &vectors,
            role,
            &tables,
            &rows,
            &MinicoilConstants::default(),
        )
        .expect("direct conversion");
        let actual =
            super::embedder::sparse_from_tokens(&fixture.tokens, &vectors, role, &tables, &rows)
                .expect("token-output glue");
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn token_output_glue_names_wrong_vector_count() {
    let error = super::embedder::sparse_from_tokens(
        &["[CLS]".to_string(), "[SEP]".to_string()],
        &[0.0; 512],
        Role::Question,
        &tables(),
        &ProjectionRows::default(),
    )
    .expect_err("one vector for two tokens");
    assert!(matches!(
        error,
        MinicoilError::TokenVectors {
            tokens: 2,
            values: 512,
            expected: 1024,
        }
    ));
}

#[test]
fn projection_row_insert_refuses_wrong_width_without_panicking() {
    let result = std::panic::catch_unwind(|| ProjectionRows::default().insert(17, vec![0.0; 2047]));
    assert!(
        result.is_ok(),
        "an invalid projection row must return an error"
    );
}

#[test]
fn local_minicoil_construction_refuses_catalog_before_files() {
    let Err(error) = super::MinicoilEmbedder::from_model_dirs(
        Path::new("missing-encoder"),
        Path::new("missing-tables"),
    ) else {
        panic!("catalog-only construction must be refused");
    };
    assert!(matches!(
        error.downcast_ref::<crate::error::TesseraError>(),
        Some(crate::error::TesseraError::ConfigError(message))
            if message.contains("minicoil-v1") && message.contains("catalog-only")
    ));
}

#[cfg(not(feature = "fetch"))]
#[test]
fn direct_minicoil_construction_refuses_catalog_before_files() {
    let Err(error) = super::MinicoilEmbedder::new("minicoil-v1") else {
        panic!("catalog-only construction must be refused");
    };
    assert!(matches!(
        error.downcast_ref::<crate::error::TesseraError>(),
        Some(crate::error::TesseraError::ConfigError(message))
            if message.contains("minicoil-v1") && message.contains("catalog-only")
    ));
}
