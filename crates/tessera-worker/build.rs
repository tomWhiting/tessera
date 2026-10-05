//! Captures source identity before the worker is compiled.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(arguments)
        .output()
        .map_err(|error| format!("cannot run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed with {}",
            arguments[0], output.status
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("git returned non-UTF-8 metadata: {error}"))
}

fn watch(root: &Path, path: &str) {
    let path = root.join(path);
    println!("cargo:rerun-if-changed={}", path.display());
}

fn source_commit(root: &Path) -> Result<String, String> {
    if !root.join(".git").exists() {
        watch(root, ".git/HEAD");
        return Err("source tree has no Git metadata".into());
    }
    if root.join(".git").is_file() {
        watch(root, ".git");
    }
    for member in ["HEAD", "index", "packed-refs"] {
        let path = git(root, &["rev-parse", "--git-path", member])?;
        watch(root, path.trim());
    }
    let reference = git(root, &["rev-parse", "--symbolic-full-name", "HEAD"])?;
    let reference = reference.trim();
    if reference != "HEAD" {
        let path = git(root, &["rev-parse", "--git-path", reference])?;
        watch(root, path.trim());
    }
    let tracked = git(root, &["ls-files", "-z"])?;
    for path in tracked.split('\0').filter(|path| !path.is_empty()) {
        watch(root, path);
    }
    let head = git(root, &["rev-parse", "--verify", "HEAD"])?;
    let head = head.trim();
    if head.len() < 12 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("Git HEAD is not a hexadecimal object id".into());
    }
    let status = git(
        root,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
    )?;
    let changes = if status.is_empty() { "" } else { "+changes" };
    Ok(format!("{}{changes}", &head[..12]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest.join("../..").canonicalize()?;
    let commit = match source_commit(&root) {
        Ok(commit) => commit,
        Err(reason) => {
            println!("cargo:warning=worker source commit unknown: {reason}");
            "unknown".to_string()
        }
    };
    let name = env::var("CARGO_PKG_NAME")?;
    let version = env::var("CARGO_PKG_VERSION")?;
    let compute = if env::var_os("CARGO_FEATURE_ACCELERATE").is_some() {
        "accelerate"
    } else {
        "plain"
    };
    println!("cargo:rustc-env=TESSERA_WORKER_BUILD={name} {version} commit {commit} {compute}");
    println!("cargo:rustc-env=TESSERA_SOURCE_COMMIT={commit}");
    Ok(())
}
