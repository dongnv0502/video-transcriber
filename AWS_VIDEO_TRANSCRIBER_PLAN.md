# AWS Video Transcriber — Implementation Plan

## Context

Build the CLI described in `aws-video-transcriber-spec.md` (repo root, Vietnamese): a batch pipeline that turns ~200 Vietnamese AWS course MP4s into timestamped, auditable transcripts (`raw.json`, `corrected.json`, `transcript.md/.txt/.srt`) with SQLite state, resume, logging, and a benchmark harness. ASR runs 100% locally on Apple Silicon (MLX Whisper `large-v3`).

**One explicit deviation from the spec, decided by the user in this session:** the correction LLM is NOT local. It is an **OpenAI-compatible HTTP endpoint** configured by `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` (may point at a remote provider or at a local server such as Ollama's `/v1`). This overrides spec §3.1 and the §18 "Privacy" checkbox "Không gọi cloud inference API". Consequences that MUST be implemented: only *flagged segment text plus one neighbouring segment on each side* ever leaves the machine — never audio, never video, never a full transcript; `transcriber correct` prints the endpoint host and the number of segments to be sent before starting; `LLM_ENABLED=false` and `--dry-run` fully bypass the network; the API key is never logged or written into artifacts.

LLM correction runs **only on flagged segments** (user decision), not on every segment.

End state: `transcriber process ./videos` scans, extracts, transcribes, corrects, and exports; a crash resumes without redoing completed work; `transcriber bench` produces a quality/perf report on 3 sample videos before the 200-video run.

## Verified environment (probed this session)

- macOS 26.5.2, arm64, Apple M4 Pro, 24 GiB unified memory, 172 GiB free disk.
- `/opt/homebrew/bin/ffmpeg`, `ffprobe`, `uv 0.9.17`, `git` present. Ollama **not** installed (not needed under the chosen LLM design).
- System `python3` is 3.14.6. **Pin the project to CPython 3.12.13** (already installed at `/opt/homebrew/bin/python3.12`): cp312 arm64 wheels verified to exist for `mlx`, `torch`, `numba`, `llvmlite`, `ctranslate2`, `onnxruntime`, `scipy`, `rapidfuzz`; cp314 coverage is incomplete for that set.
- Repo root `/Users/dongnv/Workspace/untitled folder` contains only the spec, is **not** a git repo yet, and **contains a space** — every shell snippet must quote paths.
- Verified upstream APIs (do not re-derive):
  - `mlx_whisper.transcribe(audio, *, path_or_hf_repo, verbose, temperature, compression_ratio_threshold, logprob_threshold, no_speech_threshold, condition_on_previous_text, initial_prompt, word_timestamps, clip_timestamps, hallucination_silence_threshold, **decode_options)`; `audio` accepts a float32 numpy array at 16 kHz. Returns `dict(text=..., segments=[...], language=...)`; each segment dict has `seek, start, end, text, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob` (plus `words` with `word/start/end/probability` when `word_timestamps=True`). `mlx_whisper.transcribe.ModelHolder` caches the loaded model in-process across calls, so per-window calls do not reload weights.
  - `faster_whisper` 1.2.1 ships `faster_whisper/assets/silero_vad_v6.onnx` and `faster_whisper.vad` exposing `VadOptions(threshold, neg_threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms, speech_pad_ms)`, `get_speech_timestamps(audio: np.ndarray, vad_options=None, sampling_rate=16000) -> list[{"start": samples, "end": samples}]`, `collect_chunks`, `SpeechTimestampsMap`. Its deps (`ctranslate2`, `onnxruntime`, `av`, `tokenizers`) do **not** include torch. This is the VAD we reuse — do not add `silero-vad` (it drags torch+torchaudio) and do not write a VAD from scratch.
  - HF model id for ASR: `mlx-community/whisper-large-v3-mlx` (exists, most-downloaded large-v3 MLX conversion). Benchmark comparators: `mlx-community/whisper-large-v3-turbo`, `mlx-community/whisper-large-v3-mlx-8bit`.

## Approach

### 1. Project scaffold, config, logging, DB

Create at the repo root (no subdirectory):

```
pyproject.toml   .python-version   .gitignore   .env.example   config.example.yml   README.md
glossary/aws.yml   glossary/corrections.yml
src/transcriber/{__init__,cli,config,db,logging_setup,paths,models,progress}.py
src/transcriber/stages/{__init__,scan,extract,audio,vad,transcribe,suspicious,rules,correct,export}.py
src/transcriber/llm/{__init__,client,prompt}.py
src/transcriber/benchmark.py
tests/{conftest.py,unit/,integration/,fixtures/}
```

`pyproject.toml`: `[project] name="aws-video-transcriber"`, `requires-python=">=3.12,<3.13"`, `[project.scripts] transcriber="transcriber.cli:app"`, hatchling backend with `packages=["src/transcriber"]`. Dependencies, exact floors: `mlx-whisper>=0.4.3`, `faster-whisper>=1.2.1`, `typer>=0.15`, `rich>=13`, `pydantic>=2.9`, `pydantic-settings>=2.6`, `pyyaml>=6`, `rapidfuzz>=3.10`, `httpx>=0.28`, `numpy>=1.26`. Dev extra: `pytest>=8`. No `srt`, no `jsonschema`, no `soundfile` — SRT formatting is ~15 lines, JSON validation is pydantic, WAV reading is stdlib `wave` + `numpy.frombuffer`.

`.python-version` = `3.12`. Bootstrap with `uv venv --python 3.12 && uv sync` (or `uv pip install -e ".[dev]"`).

`config.py` — `Settings(BaseSettings)` with `model_config = SettingsConfigDict(env_prefix="TRANSCRIBER_", env_file=".env", extra="ignore")`, plus a classmethod `Settings.load(config_path: Path | None, **cli_overrides)` implementing precedence **CLI flag > environment > `config.yml` > default** (read YAML into a dict, pass as init kwargs; env still wins because pydantic-settings sources rank env above init? it does not — so implement explicitly: start from YAML dict, drop keys present in `os.environ` as `TRANSCRIBER_<KEY>`, then apply CLI overrides last). Fields and defaults:

| field | default | notes |
|---|---|---|
| `video_dir` | `./videos` | |
| `output_dir` | `./output` | |
| `work_dir` | `./work` | audio cache `work/audio/<slug>.wav` |
| `db_path` | `./state.db` | |
| `log_dir` | `./logs` | |
| `model_name` | `mlx-community/whisper-large-v3-mlx` | |
| `model_backend` | `mlx` | `mlx` \| `faster-whisper` |
| `language` | `vi` | |
| `initial_prompt` | `"Bài giảng AWS tiếng Việt. Thuật ngữ: IAM, EC2, S3, VPC, CloudFormation, Lambda, ECS, EKS, CloudWatch."` | |
| `word_timestamps` | `false` | true enables numba DTW + per-word probabilities |
| `vad_enabled` | `true` | |
| `vad_threshold` | `0.5` | |
| `vad_min_silence_ms` | `700` | |
| `vad_speech_pad_ms` | `300` | |
| `vad_min_speech_ms` | `250` | |
| `vad_merge_gap_s` | `1.0` | |
| `vad_max_window_s` | `300.0` | |
| `keep_audio` | `true` | |
| `llm_enabled` | `true` | |
| `llm_base_url` | `null` | e.g. `https://api.openai.com/v1` or `http://127.0.0.1:11434/v1` |
| `llm_api_key` | `null` | from env/`.env` only; never written to YAML examples with a real value |
| `llm_model` | `null` | |
| `llm_temperature` | `0.0` | |
| `llm_batch_size` | `6` | flagged segments per request |
| `llm_response_format` | `json_schema` | `json_schema` \| `json_object` |
| `llm_timeout_s` | `120.0` | |
| `llm_max_retries` | `5` | |
| `asr_workers` | `1` | spec §13 |
| `llm_workers` | `4` | raised from spec's 1 because the LLM is now network-bound, not memory-bound; still config-capped |

If `llm_enabled` and any of `llm_base_url`/`llm_model` is unset → fail fast at command start with `ConfigError: set TRANSCRIBER_LLM_BASE_URL and TRANSCRIBER_LLM_MODEL, or run with --no-llm`. Never invent a default model name.

`logging_setup.py` — `setup_logging(log_dir, verbose: bool)`: `RotatingFileHandler(log_dir/"app.log", maxBytes=10_000_000, backupCount=5)` at DEBUG, `rich.logging.RichHandler` at INFO (DEBUG with `--verbose`). Formatter: `%(asctime)s %(levelname)s %(name)s video=%(video)s stage=%(stage)s attempt=%(attempt)s %(message)s`; provide `stage_logger(logger, video, stage, attempt)` returning a `LoggerAdapter` that fills those keys (defaults `-`). **Transcript text MUST NOT be logged above DEBUG** — enforce by only ever passing counts/ids at INFO.

`paths.py`:
- `slugify(stem: str) -> str`: NFC-normalize, lowercase, replace any run of non `[a-z0-9]` with `-`, strip leading/trailing `-`, collapse repeats, truncate to 80 chars; empty → `video`.
- `atomic_write_text(path, text)` / `atomic_write_json(path, obj)`: write `path.with_suffix(path.suffix + ".tmp")` in the same directory, `f.flush()`, `os.fsync(f.fileno())`, close, `os.replace(tmp, path)`, then fsync the parent directory fd. Every artifact write in the project goes through these two functions.

`db.py` — stdlib `sqlite3`, `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=FULL`, `PRAGMA foreign_keys=ON`, `PRAGMA user_version=1`. Schema created idempotently at startup:

```sql
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    size_bytes INTEGER,
    mtime REAL,
    duration_seconds REAL,
    status TEXT NOT NULL,
    failed_stage TEXT,
    model TEXT,
    backend TEXT,
    segment_count INTEGER,
    suspicious_count INTEGER,
    needs_review_count INTEGER,
    corrected_count INTEGER,
    llm_tokens_in INTEGER NOT NULL DEFAULT 0,
    llm_tokens_out INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT, completed_at TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_history (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage TEXT NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS llm_cache (
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (video_id, segment_id, request_hash)
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
```

Status values (string constants in `db.py`, `Status` enum): `PENDING, EXTRACTING, TRANSCRIBING, TRANSCRIBED, CORRECTING, COMPLETED, NEEDS_REVIEW, FAILED`. `TRANSCRIBED` is added to the spec list because transcribe and correct are separate reruns. Terminal success = `COMPLETED` (exports written, zero `needs_review` segments) or `NEEDS_REVIEW` (exports written, ≥1 segment flagged). `FAILED` always carries `failed_stage ∈ {extract, transcribe, correct, export}` and `error`.
Helper `set_status(conn, video_id, status, **fields)` writes `updated_at=datetime.now(timezone.utc).isoformat()` and appends a `stage_history` row; all stage runners wrap work in `try/except Exception` → `FAILED` + logged exception, never leaving a `*ING` status behind.

### 2. `scan` and `doctor`

`stages/scan.py::scan(conn, settings, rescan: bool) -> ScanReport`: `sorted(Path(video_dir).rglob("*"))` filtered to suffix in `{".mp4", ".m4v", ".mov"}` case-insensitively, skipping dotfiles and `._` AppleDouble files. For each: `path` stored as absolute POSIX string; `slug = slugify(path.stem)`, on collision append `-2`, `-3`, … until unique in DB. Insert new rows as `PENDING`. Existing row whose `size_bytes`/`mtime` changed: with `--rescan`, reset to `PENDING`, clear `error/failed_stage`, and delete its `llm_cache` rows; without `--rescan`, log a WARNING and leave it. `duration_seconds` comes from `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 -- <path>` (subprocess, no shell). Report prints counts added/updated/unchanged.

`cli.py::doctor`: prints and checks — `platform.mac_ver()`, `platform.machine()=="arm64"`, `sys.version`, `shutil.which("ffmpeg"/"ffprobe")`, `sysctl hw.memsize` via `os.sysconf`/`subprocess`, `import mlx.core` success + `mlx.core.metal.is_available()` (guard with `getattr`, tolerate API absence), presence of the model snapshot in the HF cache (`huggingface_hub.try_to_load_from_cache` or `snapshot_download(..., local_files_only=True)` in try/except), free disk on `output_dir`, and — only if `llm_enabled` — a `GET {llm_base_url}/models` with a 10 s timeout reporting status code and host (never the key). Exit code 1 if ffmpeg or mlx is missing.

### 3. Audio extraction + loading

`stages/extract.py::extract_audio(video_path, out_wav, force) -> Path`: skip when `out_wav` exists, is non-empty, and `force` is false. Command (list argv, `check=True`, `stderr=PIPE`, no shell):
`ffmpeg -nostdin -y -i <video> -vn -sn -dn -ac 1 -ar 16000 -c:a pcm_s16le -f wav <tmp>` then `os.replace(tmp, out_wav)`. On `CalledProcessError`, raise `StageError("extract", stderr.decode()[-2000:])`.

`stages/audio.py::load_wav_mono16k(path) -> np.ndarray`: stdlib `wave`; assert `getnchannels()==1`, `getframerate()==16000`, `getsampwidth()==2`; `np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0`. (Justification for not reusing `faster_whisper.audio.decode_audio`: it re-decodes through PyAV, while we already guarantee the WAV format via ffmpeg; stdlib avoids a second decode path.)

### 4. VAD windowing

`stages/vad.py::speech_windows(audio: np.ndarray, settings) -> list[Window]` where `Window = NamedTuple(start_s: float, end_s: float, start_sample: int, end_sample: int)`:

1. If `not settings.vad_enabled` or `audio.size == 0` → single window covering the whole array.
2. `from faster_whisper.vad import VadOptions, get_speech_timestamps`; `opts = VadOptions(threshold=settings.vad_threshold, min_speech_duration_ms=settings.vad_min_speech_ms, min_silence_duration_ms=settings.vad_min_silence_ms, speech_pad_ms=settings.vad_speech_pad_ms)` (leave `max_speech_duration_s` at its `inf` default; window capping happens below).
3. Convert sample chunks to seconds; **merge** consecutive chunks while `next.start - cur.end < vad_merge_gap_s` and the merged duration `<= vad_max_window_s`. A single chunk longer than `vad_max_window_s` becomes its own window (Whisper's internal 30 s sliding handles it).
4. Drop windows shorter than 0.30 s. If the result is empty → fall back to one whole-file window and log a WARNING.

Windows are **contiguous slices of the original audio**, never concatenated across silences — so segment timestamps only need `+ window.start_s`, and no `SpeechTimestampsMap` remapping is required (that class exists for the concatenated approach; we deliberately avoid it).

### 5. ASR backends + `transcribe` stage

`stages/transcribe.py` defines `class Backend(Protocol): name: str; def transcribe_window(self, audio: np.ndarray, settings) -> list[dict]` and `get_backend(name, settings) -> Backend`.

`MlxBackend.transcribe_window` calls
`mlx_whisper.transcribe(audio, path_or_hf_repo=settings.model_name, language=settings.language, initial_prompt=settings.initial_prompt, word_timestamps=settings.word_timestamps, condition_on_previous_text=False, verbose=None)` and returns `result["segments"]`.
`condition_on_previous_text=False` is deliberate: it is the standard mitigation for Whisper repetition loops across windows, and cross-window context is meaningless once VAD has cut silence.

`FasterWhisperBackend` (benchmark comparator only) wraps `WhisperModel(settings.model_name_fw or "large-v3", device="cpu", compute_type="int8")` and `model.transcribe(audio, language=..., vad_filter=False, word_timestamps=...)`, mapping its `Segment` objects to the same dict keys.

`transcribe_video(conn, settings, row, force)`:
- Skip when `output/<slug>/raw.json` exists and parses as a valid `RawTranscript` and not `force`; set status from artifact.
- `EXTRACTING` → extract (or reuse cached WAV) → `TRANSCRIBING` → load audio → `speech_windows` → for each window call the backend, offset `start`/`end` by `window.start_s`, clamp `end` to `window.end_s`, drop segments whose stripped text is empty, assign sequential `id`, record `window_index`.
- Compute `confidence`: `None` unless `word_timestamps` is on, in which case the mean of `word["probability"]`. **Never synthesize a confidence value** (spec §6.2).
- Write `raw.json` atomically; set `TRANSCRIBED`, `segment_count`, `model`, `backend`, `duration_seconds` (from ffprobe or last window end), and store `processing_seconds` + `real_time_factor` in the artifact.
- If `keep_audio` is false, delete the cached WAV after success.
- `raw.json` is written exactly once and never rewritten by any later stage (spec §3.2).

`models.py` (pydantic v2, `extra="forbid"`) with exact field names:

```python
class RawSegment(BaseModel):
    id: int; start: float; end: float; text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    confidence: float | None = None
    window_index: int = 0

class TranscriptionMeta(BaseModel):
    model: str; backend: str; language: str
    word_timestamps: bool; vad: dict
    created_at: str; tool_version: str
    processing_seconds: float; real_time_factor: float | None

class VideoMeta(BaseModel):
    filename: str; relative_path: str; duration_seconds: float | None; size_bytes: int | None

class RawTranscript(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    video: VideoMeta; transcription: TranscriptionMeta; segments: list[RawSegment]

class CorrectedSegment(RawSegment):
    raw_text: str
    corrected: bool = False
    needs_review: bool = False
    flags: list[str] = []
    correction: CorrectionInfo | None = None

class CorrectionInfo(BaseModel):
    source: Literal["glossary_case", "rule", "llm"]
    reason: str
    confidence: float | None = None
    model: str | None = None
    rejected_reason: str | None = None

class CorrectedTranscript(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    video: VideoMeta; transcription: TranscriptionMeta
    correction: CorrectionMeta; segments: list[CorrectedSegment]
```

`CorrectionMeta`: `enabled: bool, llm_model: str | None, llm_endpoint_host: str | None (host only, never the key or full URL with credentials), glossary_version: int, corrections_version: int, created_at: str, stats: dict` where stats holds `flagged, sent_to_llm, applied_llm, applied_rule, applied_case, rejected, needs_review, tokens_in, tokens_out`.

### 6. Export stage (completes the vertical slice)

`stages/export.py::export_video(settings, row, transcript: RawTranscript | CorrectedTranscript, force)` writes three files atomically into `output/<slug>/`:

- `transcript.srt`: 1-based index; `format_srt_time(t) -> "HH:MM:SS,mmm"` (`divmod`, zero-padded, milliseconds from `round(t*1000)`); blank line between cues; text on one line with internal newlines replaced by spaces.
- `transcript.txt`: segment texts joined by `"\n"`, trailing newline. No timestamps, no header.
- `transcript.md`, exactly the spec §16 shape plus a metadata block:

```markdown
# 001 Introduction

- source: `001-introduction.mp4`
- duration: 00:46:12
- model: mlx-community/whisper-large-v3-mlx
- language: vi
- segments: 384
- needs_review: 7

## Transcript

[00:02:12 - 00:02:18]
IAM Role cho phép EC2 truy cập S3
```

Title = slug with `-`/`_` → spaces, title-cased on the first character of each token only (never lowercase existing uppercase, so `IAM` survives). Timestamp line format `[HH:MM:SS - HH:MM:SS]`. For a segment with `needs_review=true`, append ` <!-- needs_review -->` to the timestamp line (invisible when rendered, greppable by agents). Export source = `corrected.json` when it exists, otherwise `raw.json`.

At this point `scan → transcribe → export` is a working vertical slice; validate it before starting §7 (spec §23.3).

### 7. Glossary, deterministic rules, suspicious detection

`glossary/aws.yml`:
```yaml
version: 1
terms: [IAM, IAM Role, IAM Policy, EC2, S3, VPC, EBS, EFS, ALB, NLB, CloudFront, CloudWatch,
        CloudFormation, Route 53, Lambda, API Gateway, ECS, EKS, Fargate, RDS, DynamoDB,
        Auto Scaling, Security Group, Subnet, Availability Zone, Region, SNS, SQS, KMS, STS]
```

`glossary/corrections.yml`:
```yaml
version: 1
corrections:
  - from: "i am role"
    to: "IAM Role"
    match: literal        # literal | regex ; literal ⇒ case-insensitive, word-boundary anchored
    auto: true            # apply deterministically when context passes
    requires_context: ["aws", "ec2", "s3", "policy", "chính sách", "quyền", "truy cập", "role"]
  - from: "i am policy"
    to: "IAM Policy"
    match: literal
    auto: true
    requires_context: ["aws", "iam", "quyền", "policy"]
  - from: "ét ba"
    to: "S3"
    match: literal
    auto: false           # candidate only ⇒ flag for the LLM, never auto-applied
    requires_context: ["bucket", "lưu trữ", "storage", "object"]
```
Ship at least 15 entries covering the service names most often mangled in Vietnamese ASR (`i am` → IAM, `e c hai`/`ec hai` → EC2, `vi pi si` → VPC, `lam đa` → Lambda, `cờ lao phót` → CloudFront, `rao 53` → Route 53, …). `requires_context` is matched over the 3-segment window (`prev + cur + next`), lowercased and diacritic-stripped on both sides via a shared `fold(s)` helper (`unicodedata.normalize("NFD", s)` minus combining marks, lowercased).

`stages/rules.py`:
- `load_glossary(path)` / `load_corrections(path)` → pydantic models with `version`.
- `apply_case_normalization(text, terms) -> (text, applied: list[str])`: for each glossary term, regex `(?<![\w])` + escaped term with `\s+` between words + `(?![\w])`, `re.IGNORECASE`; if the matched span differs from the canonical term **only by letter case** (compare `matched.casefold() == term.casefold()` and `matched != term`), replace with the canonical term. Never changes letters, digits, or spacing. Source tag `glossary_case`.
- `apply_auto_rules(prev, text, next, corrections) -> (text, applied: list[RuleHit])`: only entries with `auto: true` and a satisfied `requires_context`. Literal entries compile to `re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, from.split())) + r"(?!\w)", re.IGNORECASE)`.
- `candidate_hits(prev, text, next, corrections) -> list[str]`: entries whose pattern matches but that are not auto-applied (either `auto: false` or context unmet) — these become the `CORRECTION_CANDIDATE` flag.

`stages/suspicious.py::flag_segment(seg, prev, next, glossary, corrections, settings) -> list[str]`; any non-empty result marks the segment suspicious. Flags and thresholds (constants at module top, overridable via config later if needed):

| flag | condition |
|---|---|
| `LOW_LOGPROB` | `avg_logprob is not None and avg_logprob < -0.70` |
| `HIGH_NOSPEECH` | `no_speech_prob is not None and no_speech_prob > 0.60` |
| `HIGH_COMPRESSION` | `compression_ratio is not None and compression_ratio > 2.4` |
| `REPETITION` | `re.search(r"(?i)\b(\w+)\b(\s+\1\b){3,}", text)` |
| `FAST_TEXT` | `len(text) / max(end - start, 0.01) > 25` |
| `SHORT_SEGMENT` | `end - start < 0.35 and len(text.strip()) > 12` |
| `GLOSSARY_NEAR_MISS` | any 1–3-token n-gram `g` of the folded text with `78 <= rapidfuzz.fuzz.ratio(g, fold(term)) < 100` for some glossary term, and `g` is not exactly equal to any folded term |
| `CORRECTION_CANDIDATE` | `candidate_hits(...)` non-empty |

`suspicious_count` = number of segments with ≥1 flag; stored on the video row and printed in progress.

### 8. LLM client + `correct` stage

`llm/client.py::LlmClient` — plain `httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout_s, headers={"Authorization": f"Bearer {key}"} if key else {})`; no vendor SDK. One method:

```python
def complete(self, messages: list[dict], schema: dict, max_tokens: int) -> tuple[dict, TokenUsage]
```
POST `/chat/completions` with `{"model": settings.llm_model, "messages": ..., "temperature": settings.llm_temperature, "max_tokens": ..., "response_format": rf}` where `rf = {"type": "json_schema", "json_schema": {"name": "corrections", "strict": True, "schema": schema}}` for `llm_response_format == "json_schema"`, else `{"type": "json_object"}`. On HTTP 400/422 whose body mentions `response_format` or `json_schema`, log a WARNING, permanently downgrade this client instance to `{"type": "json_object"}`, and retry once — many OpenAI-compatible servers lack strict schema support. Parse `choices[0].message.content` with `json.loads`; on `JSONDecodeError` attempt a single salvage of the outermost `{...}` span, else raise `LlmProtocolError`. Return `usage.prompt_tokens`/`completion_tokens` (0 when absent).
Retries: on `httpx.TimeoutException`, `httpx.TransportError`, HTTP 429, and HTTP ≥500 → sleep `min(2**attempt, 16) * (1 + random.random()*0.25)` seconds, honoring an integer `Retry-After` header when present, up to `llm_max_retries`; then raise `LlmUnavailable`. Never retry 401/403 — raise immediately with "check TRANSCRIBER_LLM_API_KEY" and no key material in the message.

`llm/prompt.py` — system message (Vietnamese + English, pinned literal in code):
> Bạn là bộ hiệu đính ASR cho bài giảng AWS tiếng Việt. Chỉ được sửa lỗi nhận dạng giọng nói và chuẩn hóa thuật ngữ AWS (tên service, acronym, lệnh CLI). TUYỆT ĐỐI KHÔNG: tóm tắt, diễn giải lại, rút gọn, dịch, thêm thông tin mới, thay đổi con số. Nếu không chắc chắn: trả về changed=false và needs_review=true. Giữ nguyên văn phong và dấu câu của người giảng.

User message = compact JSON (`json.dumps(..., ensure_ascii=False)`):
`{"glossary": [...≤40 terms relevant to this batch...], "items": [{"id": 12, "prev": "...", "text": "...", "next": "...", "flags": ["GLOSSARY_NEAR_MISS"]}]}` — `prev`/`next` are context only and MUST NOT be returned; state that in the system message. Batch size `llm_batch_size` (6). `max_tokens = 220 * len(items)`.

Response schema (the exact object passed as `schema`):
```json
{"type":"object","additionalProperties":false,"required":["results"],
 "properties":{"results":{"type":"array","items":{
   "type":"object","additionalProperties":false,
   "required":["id","changed","corrected","reason","confidence","needs_review"],
   "properties":{"id":{"type":"integer"},"changed":{"type":"boolean"},
     "corrected":{"type":["string","null"]},"reason":{"type":"string"},
     "confidence":{"type":"number"},"needs_review":{"type":"boolean"}}}}}}
```

`stages/correct.py::correct_video(conn, settings, row, dry_run, force)`:

1. Load `raw.json` (fail → `FAILED/correct`). Set `CORRECTING`.
2. Build `CorrectedSegment` list from raw with `raw_text = text`.
3. Deterministic pass per segment, in order: `apply_case_normalization`, then `apply_auto_rules`. Any applied change sets `corrected=true` and `correction=CorrectionInfo(source="glossary_case"|"rule", reason=...)`. Deterministic changes never set `needs_review`.
4. Flag pass: `flags = flag_segment(...)` computed on the post-deterministic text.
5. If `llm_enabled` and not `dry_run`: collect flagged segments, split into batches of `llm_batch_size`, run over a `ThreadPoolExecutor(max_workers=llm_workers)` (batches are independent; results are merged by `id`, so ordering is unaffected). Before the first request, print `LLM: <n> segment(s) → <host> (<model>)`. Per segment, `request_hash = sha256(model + prompt_version + prev + text + next + sorted(flags))`; a matching `llm_cache` row short-circuits the call, and every fresh response is inserted into `llm_cache` (this is the resume unit — a re-run of `correct` after a crash re-sends nothing already answered).
6. **Validation gate**, applied to each returned item before anything is accepted (this is the guard that keeps spec §3.3 true):
   1. `id` not in the request → drop, log WARNING.
   2. `changed == false` → keep raw text; `needs_review = item.needs_review`.
   3. `changed == true and (corrected is None or corrected.strip() == "")` → reject.
   4. `corrected.strip() == raw.strip()` → treat as unchanged.
   5. `confidence < 0.85` → reject.
   6. `rapidfuzz.fuzz.ratio(raw, corrected) < 60` → reject.
   7. `len(corrected) / max(len(raw), 1)` outside `[0.7, 1.4]` → reject.
   8. `re.findall(r"\d+", raw) != re.findall(r"\d+", corrected)` → reject (numbers must never change).
   9. `abs(wordcount(corrected) - wordcount(raw)) > 3` → reject (blocks paraphrase/expansion).
   Accept → `text = corrected`, `corrected = True`, `correction = CorrectionInfo(source="llm", reason, confidence, model)`, `needs_review = item.needs_review`. Reject → text stays raw, `needs_review = True`, `correction = CorrectionInfo(source="llm", reason, confidence, model, rejected_reason="<rule name>")`. Ids absent from the response → unchanged + `needs_review = True`.
7. Segments still carrying flags but never sent (LLM disabled or dry run) → `needs_review = True` with `correction = None`.
8. Write `corrected.json` and `corrections.json` atomically. `corrections.json` is the structured diff required by spec §18: `{"schema_version":"1.0","video":"<slug>","stats":{...},"changes":[{"id":12,"start":121.42,"end":127.31,"raw":"...","corrected":"...","source":"llm","reason":"...","confidence":0.98,"accepted":true,"rejected_reason":null,"flags":[...]}]}` — one entry per segment that was changed **or** rejected **or** marked `needs_review`.
9. Update row: `corrected_count`, `needs_review_count`, `suspicious_count`, token counters; status `TRANSCRIBED` → export handles the rest; the final status (`COMPLETED` vs `NEEDS_REVIEW`) is set by the export stage from `needs_review_count`.
10. On `LlmUnavailable` after retries → `FAILED/correct`; `raw.json` untouched (spec §20).

`--dry-run` performs steps 1–4 and 7 and prints the batch/segment counts and the target host, writes nothing, and leaves status unchanged.

### 9. Orchestration: `process`, `resume`, `status`, retries, progress

`cli.py` (Typer app, one command per spec §11): `scan`, `transcribe`, `correct`, `export`, `status`, `resume`, `process`, `bench`, `doctor`. Global options: `--config PATH`, `--verbose`, `--limit N`, `--video PATH`, `--force`, `--no-llm`.

Work selection queries (exact):
- `transcribe`: `status IN ('PENDING','EXTRACTING','TRANSCRIBING')` or (`status='FAILED' AND failed_stage IN ('extract','transcribe')`), ordered by `path`. `--force` additionally re-runs any status for the selected videos and rewrites `raw.json`.
- `correct`: `status='TRANSCRIBED'` or (`status='FAILED' AND failed_stage='correct'`); `--force` also re-runs `COMPLETED`/`NEEDS_REVIEW`.
- `export`: any video with a readable `raw.json`; `--force` overwrites existing exports, otherwise skips when all three files exist and are newer than their source JSON.
- `resume` = `transcribe` then `correct` then `export` over their respective selections. `process DIR` = `scan DIR` then `resume`.

Failure isolation: a per-video exception is caught by the batch loop, recorded (`attempts += 1`, `error`, `failed_stage`, `stage_history`), logged with video path + stage + attempt + exception, and the loop continues to the next video. A `KeyboardInterrupt` marks the current video back to its last durable status (`PENDING` or `TRANSCRIBED`) and exits 130.

`progress.py` renders exactly the spec §14 lines via `rich`:
```
[037/200] 001-introduction
progress: 82%
segments: 384
suspicious: 7
status: transcribing
```
and on completion:
```
completed
output: output/001-introduction/
duration: 46m12s
processing: 18m31s
segments: 384
needs_review: 7
```
`progress: N%` during transcription = processed window seconds / total window seconds.

`status` prints a rich table of counts per status plus totals (`200 total / 137 completed / 1 failed / 62 pending`) and, with `--json`, the same as machine-readable JSON on stdout.

### 10. Benchmark harness

`benchmark.py` + `transcriber bench --videos A.mp4 B.mp4 C.mp4 [--backends mlx] [--models m1,m2] [--out benchmark/]`:

For each (video × backend × model): run extraction + VAD + transcription into a temp output root (never touching `output/`), measuring wall time, `real_time_factor = processing / duration`, peak RSS via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` sampled before/after, segment count, mean `avg_logprob`, and count of segments with each suspicious flag. GPU/Metal utilization is **not** measured (requires `sudo powermetrics`); the report prints `metal_utilization: not measured` — spec §17 allows this.

Quality scoring against optional `benchmark/ground_truth/<slug>.yml`:
```yaml
must_contain: ["IAM Role", "EC2", "S3", "CloudFormation", "aws s3 ls"]
must_not_contain: ["I am role", "ét ba"]
sample_segments:
  - start: 121.4
    text: "IAM Role cho phép EC2 truy cập S3"
```
Metrics: term recall = matched `must_contain` / total (exact, case-sensitive, over the whole transcript); false-positive count from `must_not_contain`; per-sample WER computed by matching the transcript segment whose `start` is nearest the sample `start` and running token-level `rapidfuzz.distance.Levenshtein.distance(ref_tokens, hyp_tokens) / len(ref_tokens)`. Also report correction false positives: number of accepted LLM changes on segments listed in `must_not_contain`-free ground truth is not derivable, so instead report the accept/reject breakdown from `corrections.json` when `--with-correction` is passed.

Output: `benchmark/<UTC timestamp>/results.json` + `report.md` (one table per metric, one row per run). No accuracy or speed number is ever stated anywhere in docs unless it comes from a `results.json` produced on this machine (spec §23.10).

### 11. Tests (written alongside each step, run per step)

`tests/unit/` — no models, no network, no ffmpeg:
`test_paths.py` (slugify normalization/collisions; `atomic_write_json` leaves no `.tmp` and never a partial file when the serializer raises mid-write), `test_config.py` (CLI > env > YAML > default precedence), `test_db.py` (status transitions, `stage_history` rows, resume selection queries return exactly the intended ids for a seeded mix of statuses), `test_vad.py` (window merging over synthetic chunk lists: gap merge, max-window cap, oversized single chunk, empty input fallback), `test_rules.py` (case normalization only changes case; `requires_context` gating; `auto:false` never applies), `test_suspicious.py` (table-driven, one case per flag plus a clean segment), `test_correct_validation.py` (accept/reject matrix — one case per numbered gate in §8.6, asserting text preservation and `rejected_reason`), `test_export.py` (SRT timing format including `>1h` and millisecond rounding; markdown shape incl. the `needs_review` comment), `test_llm_client.py` (hermetic `http.server` in a thread: happy path, 429 then success, `response_format` 400 → json_object downgrade, malformed JSON → `LlmProtocolError`).

`tests/integration/` — marked `@pytest.mark.integration`, opt-in via `-m integration`:
`test_extract.py` (session fixture builds a 3 s MP4 with `ffmpeg -f lavfi -i sine=frequency=440:duration=3 -f lavfi -i testsrc=duration=3:size=320x240:rate=10 -shortest`; assert the produced WAV is mono/16 kHz/16-bit via `wave`), `test_transcribe_tiny.py` (marked `requires_model`; `mlx-community/whisper-tiny` over a short spoken clip generated by macOS `say -v Linh` piped through ffmpeg; assert non-empty text, monotonic non-overlapping timestamps, ids sequential), `test_pipeline_resume.py` (seed DB with one video FAILED at `correct` and one `PENDING`; run `resume` with a stubbed backend and stubbed LLM; assert the completed video's `raw.json` mtime is unchanged and only the intended videos moved status), `test_export_e2e.py` (fixture `raw.json` → all three exports byte-compared against fixtures).

`tests/fixtures/` holds `raw_sample.json`, expected `transcript.srt/.md/.txt`, and a small `corrections.yml`.

## Critical files & anchors

- `aws-video-transcriber-spec.md` §3.2–3.3 (raw immutability, no-rewrite rules), §14 (exact CLI output shape), §16 (output tree + markdown shape), §18 (acceptance checkboxes) — the contract this plan implements.
- `.venv/lib/python3.12/site-packages/faster_whisper/vad.py` — `VadOptions`, `get_speech_timestamps` signatures; reread before writing `stages/vad.py`.
- `.venv/lib/python3.12/site-packages/mlx_whisper/transcribe.py` — `transcribe(...)` keyword list, `new_segment(...)` field names, `ModelHolder` caching; reread before writing `stages/transcribe.py`.
- `src/transcriber/stages/correct.py` — the nine-rule validation gate is the only thing preventing the LLM from rewriting lecture content; it must run on every returned item with no bypass flag.
- `src/transcriber/paths.py` — `atomic_write_*` is the single write path for all artifacts; any direct `open(..., "w")` on an artifact is a bug.

## Verification

Run everything from the repo root (path contains a space — keep it quoted).

1. **Bootstrap + static check**
   `uv venv --python 3.12 && uv pip install -e ".[dev]" && uv run transcriber --help`
   Expect: help listing `scan transcribe correct export status resume process bench doctor`.
2. **Environment gate**
   `uv run transcriber doctor`
   Expect: `arm64 ✓`, `python 3.12.x ✓`, `ffmpeg ✓`, `mlx ✓ (metal available)`, LLM line showing only the host and HTTP status. Exit 0.
3. **Unit suite** (after each step, at minimum at the end)
   `uv run pytest tests/unit -q` → all pass.
4. **Integration suite**
   `uv run pytest tests/integration -q -m integration` → all pass; the `requires_model` test downloads `whisper-tiny` once.
5. **New-behavior end-to-end, 1 real video** (the primary proof)
   ```
   mkdir -p sample && cp "<one real course mp4>" sample/
   uv run transcriber process ./sample
   ```
   Expect, concretely:
   - `output/<slug>/` contains `raw.json`, `corrected.json`, `corrections.json`, `transcript.md`, `transcript.txt`, `transcript.srt`.
   - `python -c "import json;d=json.load(open('output/<slug>/raw.json'));s=d['segments'];print(len(s), all(a['end']<=b['start']+0.05 for a,b in zip(s,s[1:])), s[0]['start']>=0)"` → segment count > 0, `True True`.
   - `raw.json` and `corrected.json` have identical `segments[i].start/end` for every `i`, and `corrected.json[i].raw_text == raw.json[i].text` for every `i` (run as a one-off assert script) — proves raw is never overwritten and timestamps are preserved.
   - `head -12 output/<slug>/transcript.md` matches the spec §16 shape; `head -6 output/<slug>/transcript.srt` shows `1`, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text, blank.
   - Manual spot check: open the video at the timestamp of segment 5 from `transcript.srt` and confirm the spoken words match (timestamp truth cannot be automated).
   - `uv run transcriber status` shows `1 total / 1 completed` (or `needs_review` with a non-zero count).
6. **Resume proof**
   Re-run `uv run transcriber process ./sample`; expect log lines `skip (raw.json present)` and unchanged `raw.json` mtime (`stat -f %m`).
   Then simulate a crash: `sqlite3 state.db "UPDATE videos SET status='FAILED', failed_stage='correct';"`, delete `corrected.json`, re-run `uv run transcriber resume`; expect only the correct+export stages to run (no re-transcription — check `stage_history`) and `llm_cache` hits reported as `cached: N` so no duplicate API calls.
7. **Privacy boundary proof**
   `uv run transcriber correct --video sample/<file>.mp4 --dry-run` → prints the flagged-segment count and target host, makes zero HTTP requests (verify with `TRANSCRIBER_LLM_BASE_URL=http://127.0.0.1:9/v1` — an unreachable port — and observe success).
   `uv run transcriber transcribe --video ... --no-llm` with the same unreachable base URL → completes, proving ASR has no network dependency after the model is cached.
   `grep -ri "<last 6 chars of api key>" output logs` → no matches.
8. **Benchmark before the 200-video run** (spec §17 gate)
   `uv run transcriber bench --videos sample/a.mp4 sample/b.mp4 sample/c.mp4 --models mlx-community/whisper-large-v3-mlx,mlx-community/whisper-large-v3-turbo`
   Expect `benchmark/<ts>/report.md` with per-run wall time, RTF, peak RSS, segment counts, and — where `benchmark/ground_truth/<slug>.yml` exists — term recall, false positives, and sample WER. Only after reading this report should the full `transcriber process ./videos` run start.

## Assumptions & contingencies

- **LLM endpoint is user-supplied.** No default `llm_base_url`/`llm_model`; commands fail fast with the exact env var names when `llm_enabled` and they are unset. *If the user later wants fully local inference*, set `TRANSCRIBER_LLM_BASE_URL=http://127.0.0.1:11434/v1` and `TRANSCRIBER_LLM_MODEL=qwen2.5:14b-instruct-q4_K_M` after `brew install ollama && ollama pull qwen2.5:14b-instruct-q4_K_M` (verified: 8.99 GB, fits 24 GiB) — no code change required, and the README must document this.
- **Videos live wherever `TRANSCRIBER_VIDEO_DIR` points**; `./videos` is only a default and does not exist yet. Do not hard-code any user path (spec §21).
- **`mlx-community/whisper-large-v3-mlx` is the ASR default.** *If the benchmark in step 8 shows RTF or Vietnamese/AWS-term accuracy is unacceptable*, switch `model_name` (config only, no code change) to `mlx-community/whisper-large-v3-turbo` for speed or `...-large-v3-mlx-8bit` for memory, and rerun the benchmark; do not change the pipeline.
- **VAD is on by default.** *If VAD windowing measurably drops speech* (compare `bench` runs with `TRANSCRIBER_VAD_ENABLED=false`), fall back to whole-file transcription via config; the window code path already handles the single-window case.
- **Python is pinned to 3.12.** *If `uv pip install` fails to resolve any wheel on 3.12*, do not switch to 3.14 — install cpython 3.13 via `uv python install 3.13`, update `.python-version` and `requires-python`, and re-verify `mlx`, `torch`, `numba`, `ctranslate2`, `onnxruntime` all resolve to arm64 wheels.
- **`llm_workers=4`** deviates from spec §13's default of 1 because the correction stage is now network-bound. *If the endpoint rate-limits* (sustained 429s in `logs/app.log`), drop to `TRANSCRIBER_LLM_WORKERS=1`; ASR stays at 1 worker regardless (MLX Metal contention).
