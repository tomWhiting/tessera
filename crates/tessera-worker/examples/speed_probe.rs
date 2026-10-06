//! Measure quiet framed queries and spawn-to-first-vector through the shared API.

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use haem_frames::embedding::{
    check_vectors, decode_vector, read_message, write_message, Embed, Input, Kind, Limits, Message,
    Outcome, Ready, Start,
};
use sha2::{Digest, Sha256};

type ProbeResult<T> = Result<T, Box<dyn Error>>;
const QUERY: &str = "where is the red book?";
const FRAME: u32 = 131_072;

struct Session {
    child: Child,
    input: Option<ChildStdin>,
    output: ChildStdout,
    ready: Ready,
    start: Start,
}

const fn limits() -> Limits {
    Limits {
        memory_bytes: 2_147_483_648,
        threads: 2,
        batch_items: 1,
        input_bytes: 65_536,
        tokens: 512,
        frame_bytes: FRAME,
    }
}

fn request() -> Embed {
    Embed {
        kind: Kind::Query,
        items: vec![Input {
            id: "query".to_string(),
            text: QUERY.to_string(),
        }],
    }
}

impl Session {
    fn spawn(binary: &Path, directory: &Path) -> ProbeResult<Self> {
        let start = Start {
            protocol: 1,
            model_dir: directory
                .to_str()
                .ok_or("probe_model_path_not_utf8")?
                .to_string(),
            limits: limits(),
            windows: None,
        };
        let mut child = Command::new(binary)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let result = (|| {
            let mut input = child.stdin.take().ok_or("probe_missing_stdin")?;
            let mut output = child.stdout.take().ok_or("probe_missing_stdout")?;
            let limit = NonZeroU32::new(FRAME).ok_or("probe_zero_frame_limit")?;
            write_message(
                &mut input,
                &Message::Start(Start {
                    protocol: start.protocol,
                    model_dir: start.model_dir.clone(),
                    limits: limits(),
                    windows: None,
                }),
                limit,
            )?;
            let Some(Message::Ready(ready)) = read_message(&mut output, limit)? else {
                return Err("probe_missing_ready".into());
            };
            if ready.model.name != "bge-base-en-v1.5"
                || ready.model.revision != "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
                || ready.model.dimensions != 768
            {
                return Err("probe_model_identity_mismatch".into());
            }
            Ok((input, output, ready))
        })();
        match result {
            Ok((input, output, ready)) => Ok(Self {
                child,
                input: Some(input),
                output,
                ready,
                start,
            }),
            Err(error) => {
                child.kill()?;
                child.wait()?;
                Err(error)
            }
        }
    }
    fn query(&mut self) -> ProbeResult<String> {
        let request = request();
        let limit = NonZeroU32::new(FRAME).ok_or("probe_zero_frame_limit")?;
        write_message(
            self.input.as_mut().ok_or("probe_input_closed")?,
            &Message::Embed(self::request()),
            limit,
        )?;
        let Some(Message::Vectors(vectors)) = read_message(&mut self.output, limit)? else {
            return Err("probe_missing_vectors".into());
        };
        check_vectors(&vectors, &request, &self.ready, &self.start.limits)?;
        let [Outcome::Vector {
            id,
            vector,
            tokens_read,
            tokens_total,
        }] = vectors.items.as_slice()
        else {
            return Err("probe_vector_shape".into());
        };
        if id != "query" || *tokens_read != 16 || *tokens_total != 16 {
            return Err("probe_query_count_or_id".into());
        }
        let values = decode_vector(vector)?;
        if values.len() != 768 || values.iter().any(|x| !x.is_finite()) {
            return Err("probe_vector_invalid".into());
        }
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    fn close(&mut self) -> ProbeResult<()> {
        drop(self.input.take());
        if !self.child.wait()?.success() {
            return Err("probe_worker_exit_failed".into());
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        drop(self.input.take());
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                if let Err(error) = self.child.kill() {
                    eprintln!("probe child kill failed: {error}");
                }
                if let Err(error) = self.child.wait() {
                    eprintln!("probe child reap failed: {error}");
                }
            }
            Err(error) => eprintln!("probe child status failed: {error}"),
        }
    }
}

fn command(program: &str, args: &[&str]) -> ProbeResult<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!("probe_metadata_command: {program}: {}", output.status).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn record(
    file: &mut impl Write,
    session: &Session,
    identity: (&str, usize),
    elapsed: Duration,
    hash: &str,
    load: &str,
    binary_hash: &str,
) -> ProbeResult<()> {
    let rss = command("ps", &["-o", "rss=", "-p", &session.child.id().to_string()])?
        .parse::<u64>()?
        .checked_mul(1024)
        .ok_or("probe_rss_overflow")?;
    let observation = serde_json::json!({"schema_version":1,"journey":identity.0,"route":"worker","dataset":"query",
        "observation":identity.1,"elapsed_ns":elapsed.as_nanos(),"elapsed_seconds":elapsed.as_secs_f64(),
        "worker_pid":session.child.id(),"worker_binary_sha256":binary_hash,"accelerate":cfg!(feature="accelerate"),
        "model":session.ready.model.name,"revision":session.ready.model.revision,"installed_manifest_sha256":session.ready.model.manifest_sha256,
        "device":"cpu","dtype":"f32","threads":2,"load_average_start":load,"sampled_worker_rss_bytes_after":rss,
        "counts":{"items":1,"vectors":1,"forward_calls_derived":1,"real_tokens":16,"physical_token_rows_derived":16,
            "real_squared_lengths":256,"physical_squared_lengths_derived":256,"raw_output_bytes":3072},
        "output_sha256":hash,"status":"passed","timing_boundary":"parent frame construction/write through shared validation/decode and output hash sink; excludes JSONL/RSS sampling"});
    serde_json::to_writer(&mut *file, &observation)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn main() -> ProbeResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: speed_probe WORKER MODEL_DIR query|startup OUTPUT_JSONL".into());
    }
    let binary = Path::new(&args[0]);
    let directory = Path::new(&args[1]);
    let mode = args[2].to_str().ok_or("probe_mode_not_utf8")?;
    if !["query", "startup"].contains(&mode) {
        return Err("probe_unknown_mode".into());
    }
    let load = command("sysctl", &["-n", "vm.loadavg"])?;
    let binary_hash = format!("{:x}", Sha256::digest(std::fs::read(binary)?));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    if mode == "query" {
        let mut session = Session::spawn(binary, directory)?;
        session.query()?;
        for observation in 0..100 {
            let start = Instant::now();
            let hash = session.query()?;
            let elapsed = start.elapsed();
            record(
                &mut file,
                &session,
                ("J2", observation),
                elapsed,
                &hash,
                &load,
                &binary_hash,
            )?;
        }
        session.close()?;
    } else {
        for observation in 0..20 {
            let start = Instant::now();
            let mut session = Session::spawn(binary, directory)?;
            let hash = session.query()?;
            let elapsed = start.elapsed();
            record(
                &mut file,
                &session,
                ("J4", observation),
                elapsed,
                &hash,
                &load,
                &binary_hash,
            )?;
            session.close()?;
        }
    }
    file.sync_all()?;
    Ok(())
}
