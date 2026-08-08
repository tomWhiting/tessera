use super::display_bytes;

#[test]
fn formats_artifact_sizes_compactly() {
    assert_eq!(display_bytes(512), "512 B");
    assert_eq!(display_bytes(1024 * 1024), "1.0 MiB");
}
