use std::error::Error;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

const MAX_LINES: usize = 500;
const SOURCE_ROOTS: &[&str] = &["src", "tests", "examples", "build_support", "xtask/src"];

pub(crate) fn check_file_sizes(repository: &Path) -> Result<(), Box<dyn Error>> {
    let mut source_files = Vec::new();
    source_files.push(repository.join("build.rs"));

    for root in SOURCE_ROOTS {
        collect_sources(&repository.join(root), &mut source_files)?;
    }

    source_files.sort();
    let mut oversized = Vec::new();
    for source_file in source_files {
        let contents = std::fs::read_to_string(&source_file)?;
        let line_count = contents.lines().count();
        if line_count > MAX_LINES {
            oversized.push((source_file, line_count));
        }
    }

    if oversized.is_empty() {
        println!("file-size policy passed: all handwritten sources are <= {MAX_LINES} lines");
        return Ok(());
    }

    for (path, line_count) in &oversized {
        eprintln!(
            "file-size policy violation: {} has {line_count} lines (maximum {MAX_LINES})",
            display_path(repository, path).display()
        );
    }

    Err(format!("{} source files exceed {MAX_LINES} lines", oversized.len()).into())
}

fn collect_sources(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if !directory.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_sources(&path, output)?;
        } else if matches!(path.extension().and_then(OsStr::to_str), Some("rs" | "py")) {
            output.push(path);
        }
    }

    Ok(())
}

fn display_path<'a>(repository: &'a Path, path: &'a Path) -> &'a Path {
    path.strip_prefix(repository).unwrap_or(path)
}
