use super::display_bytes;

#[test]
fn formats_artifact_sizes_compactly() {
    assert_eq!(display_bytes(512), "512 B");
    assert_eq!(display_bytes(1024 * 1024), "1.0 MiB");
}

#[test]
fn installed_folder_arguments_are_explicit() {
    use super::{CertCli, CertCommand};
    use clap::Parser;
    let install = CertCli::try_parse_from([
        "cert",
        "install",
        "--model",
        "bge-base-en-v1.5",
        "--dir",
        "/installed",
    ])
    .unwrap();
    assert!(
        matches!(install.command, CertCommand::Install { model, dir } if model == "bge-base-en-v1.5" && dir == std::path::Path::new("/installed"))
    );
    let run = CertCli::try_parse_from([
        "cert",
        "run",
        "--model",
        "bge-base-en-v1.5",
        "--model-dir",
        "/installed",
    ])
    .unwrap();
    assert!(
        matches!(run.command, CertCommand::Run { model_dir: Some(dir), .. } if dir == std::path::Path::new("/installed"))
    );
    assert!(CertCli::try_parse_from(["cert", "run-all", "--model-dir", "/installed"]).is_err());
}
