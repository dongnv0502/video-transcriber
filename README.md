<div align="center">

# Video Transcriber

**High-throughput, privacy-first local video transcription and terminology correction pipeline engineered for Apple Silicon.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon%20(arm64)-black.svg)](https://support.apple.com/en-us/HT211814)
[![ASR Engine](https://img.shields.io/badge/engine-MLX%20Whisper-purple.svg)](https://github.com/ml-explore/mlx-examples)
[![VAD](https://img.shields.io/badge/VAD-Silero%20ONNX-orange.svg)](https://github.com/snakers4/silero-vad)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Architecture](#architecture) •
[Quick Start](#quick-start) •
[Performance](#performance) •
[CLI Reference](#cli-reference) •
[Configuration](#configuration) •
[Output Formats](#output-formats)

</div>

---

## Overview

**Video Transcriber** is an industrial-grade, local CLI pipeline designed to process large video libraries into timestamped, auditable, and formatted transcripts. Built natively for Apple Silicon unified memory using Apple's **MLX** framework, it transcribes audio up to **6.6× faster than real-time** on Whisper `large-v3` without sending a single byte of media to external clouds.

The pipeline includes Voice Activity Detection (VAD) windowing, deterministic technical terminology standardization, automated anomaly flagging, and an optional LLM-assisted verification gate with strict anti-hallucination constraints.

### Key Highlights

- **100% Local Inference**: Zero audio or video ever leaves your workstation.
- **Metal-Accelerated ASR**: Native MLX Whisper (`large-v3`, `turbo`, `tiny`, or 8-bit quantized models).
- **Intelligent Speech Chunking**: Silero VAD windowing prevents audio hallucination and drift across long silences.
- **Transactional State & Crash Resilience**: SQLite-backed checkpoint state machine ensures seamless resume after interruptions, restarts, or crashes without duplicating work.
- **Two-Tier Terminology Normalization**:
  - Deterministic case-normalization & context-gated phonetics rules.
  - Optional OpenAI-compatible LLM verification gate (local Ollama or remote) protected by strict 9-rule rejection heuristics (word count preservation, numeric constancy, fuzzy matching limits).
- **Multi-Format Artifact Generation**: Generates `transcript.srt`, `transcript.md`, `transcript.txt`, immutable `raw.json`, and structured diff logs (`corrections.json`).

---

## Architecture

```
                    Input Video (.mp4 / .mov / .m4v)
                                   │
                                   ▼
                      [ 1. Media Scanner & Prober ]
                             (SQLite DB State)
                                   │
                                   ▼
                   [ 2. FFmpeg 16kHz PCM Extraction ]
                                   │
                                   ▼
                     [ 3. Silero VAD Segmentation ]
                        (Contiguous Windowing)
                                   │
                                   ▼
                 [ 4. MLX Whisper ASR (Apple Silicon) ]
                                   │
                                   ▼
                  Immutable raw.json (Timestamp Ground Truth)
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     [ 5. Terminology Glossary ]          [ 6. Anomaly Detector ]
     - Case normalization rules           - Compression / logprob checks
     - Context-aware phonetic rules       - Repetition / speed anomalies
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                  [ 7. Segment Correction Gate ]
             (Deterministic + Optional LLM Verification)
                                   │
                                   ▼
           [ 8. Multi-Format Exporter (Atomic Durability) ]
         ├── transcript.srt  (Subtitles)
         ├── transcript.md   (Timecoded Markdown with review tags)
         ├── transcript.txt  (Plaintext corpus)
         ├── corrected.json  (Standardized transcript)
         └── corrections.json(Auditable diff log)
```

---

## Performance

Benchmarked on **Apple M4 Pro (24 GB Unified Memory)** using `mlx-community/whisper-large-v3-mlx`:

| Metric | Real-World Performance | Notes |
|---|---|---|
| **Real-Time Factor (RTF)** | **0.152** | $\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$ |
| **Throughput Speed** | **6.57× faster than real-time** | 1 hour of video processed in ~9.1 minutes |
| **Peak Memory (RSS)** | **~2.5 GB** | Fits comfortably in any Apple Silicon Mac |
| **Acoustic Confidence** | **Mean logprob: -0.106** | Near-zero score indicates high certainty |
| **Hardware Utilization** | **100% Metal GPU** | Low CPU overhead, zero thermal throttling |

---

## Quick Start

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4, arm64).
- **Python 3.12** installed.
- **FFmpeg & FFprobe** installed (`brew install ffmpeg`).
- **[uv](https://github.com/astral-sh/uv)** package manager (`brew install uv`).

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/video-transcriber.git
cd video-transcriber

# Create environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify environment readiness
uv run transcriber doctor --no-llm
```

---

## Usage

### 1. End-to-End Batch Processing

Scan a video directory, transcribe, normalize, and export:

```bash
# Process all videos in ./videos without external LLM
uv run transcriber process ./videos --no-llm

# Process a single specific video
uv run transcriber process ./videos --video ./videos/lecture-01.mp4 --no-llm
```

### 2. Inspect Batch Status

View real-time progress and summary statistics:

```bash
uv run transcriber status

# Output as JSON for integration scripts:
uv run transcriber status --json
```

```
AWS Video Transcriber — Batch Status 
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Status       ┃ Count ┃ Percentage ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ COMPLETED    │    45 │      90.0% │
│ NEEDS_REVIEW │     5 │      10.0% │
│ FAILED       │     0 │       0.0% │
└──────────────┴───────┴────────────┘
50 total / 45 completed / 0 failed / 0 pending
```

### 3. Fault Tolerance & Resume

If execution is interrupted (system reboot, sleep, `Ctrl+C`):

```bash
uv run transcriber resume --no-llm
```
*The system checks file modification times and SQLite stage logs, instantly resuming from the exact point of interruption without re-extracting audio or re-transcribing completed videos.*

### 4. Granular Pipeline Execution

Each phase can be invoked independently for pipeline staging or debugging:

```bash
# Step 1: Scan and register media files into database
uv run transcriber scan ./videos

# Step 2: Extract audio and transcribe (writes raw.json)
uv run transcriber transcribe --limit 10

# Step 3: Preview terminology corrections without applying
uv run transcriber correct --dry-run

# Step 4: Run terminology correction (writes corrected.json & corrections.json)
uv run transcriber correct

# Step 5: Export subtitles and document artifacts
uv run transcriber export
```

### 5. Automated Benchmark Suite

Benchmark accuracy, memory, and throughput on sample videos across models:

```bash
uv run transcriber bench \
  --videos ./videos/sample-1.mp4 ./videos/sample-2.mp4 \
  --backends mlx \
  --models mlx-community/whisper-large-v3-mlx,mlx-community/whisper-large-v3-turbo
```

Detailed Markdown performance reports are automatically saved to `benchmark/<timestamp>/report.md`.

---

## Configuration

Configuration values are resolved through strict precedence:
$$\textbf{CLI Flag} > \textbf{Environment Variable} > \textbf{config.yml} > \textbf{Defaults}$$

Copy the example template to get started:

```bash
cp config.example.yml config.yml
```

### Key Options (`config.yml`)

```yaml
# Media directories
video_dir: "./videos"
output_dir: "./output"
work_dir: "./work"
db_path: "./state.db"
log_dir: "./logs"

# Whisper ASR Configuration
model_name: "mlx-community/whisper-large-v3-mlx"  # or whisper-large-v3-turbo
model_backend: "mlx"                              # mlx | faster-whisper
language: "vi"                                    # vi, en, ja, zh, etc.
word_timestamps: false
keep_audio: true

# Voice Activity Detection (VAD)
vad_enabled: true
vad_threshold: 0.5
vad_min_silence_ms: 700
vad_speech_pad_ms: 300
vad_max_window_s: 300.0

# LLM Correction Gate (Optional)
llm_enabled: false
llm_base_url: "http://127.0.0.1:11434/v1"         # Local Ollama or remote endpoint
llm_model: "qwen2.5:14b-instruct-q4_K_M"
llm_temperature: 0.0
llm_batch_size: 6
llm_response_format: "json_schema"                # json_schema | json_object
```

### Environment Variables

All settings can be overridden via `TRANSCRIBER_<KEY>`:

```bash
export TRANSCRIBER_MODEL_NAME="mlx-community/whisper-large-v3-mlx"
export TRANSCRIBER_LANGUAGE="en"
export TRANSCRIBER_LLM_ENABLED="false"
```

---

## Terminology Dictionaries

Standardize specialized domain terms, acronyms, and phonetic misrecognitions using custom YAML dictionaries located in `glossary/`:

### Canonical Terms (`glossary/aws.yml`)
Case-normalizes recognized terms without altering content:
```yaml
version: 1
terms:
  - API Gateway
  - CloudFront
  - DynamoDB
  - Kubernetes
  - PostgreSQL
```

### Contextual Rules (`glossary/corrections.yml`)
Fixes common ASR phonetic substitutions based on adjacent context keywords:
```yaml
version: 1
corrections:
  - from: "cờ lao phót"
    to: "CloudFront"
    match: literal
    auto: true
    requires_context: ["cdn", "cache", "edge", "distribution"]

  - from: "ét ba"
    to: "S3"
    match: literal
    auto: false   # Candidate only: flagged for review or LLM verification
    requires_context: ["storage", "bucket", "object"]
```

---

## Output Formats

All artifacts are generated atomically in isolated directories `output/<video-slug>/`:

| File | Format | Description |
|---|---|---|
| `transcript.srt` | SubRip Text | Standard subtitle cues with millisecond accuracy (`HH:MM:SS,mmm`). |
| `transcript.md` | Markdown | Structured document with timecodes `[HH:MM:SS - HH:MM:SS]` and review tags. |
| `transcript.txt` | Plain Text | Clean contiguous text corpus suitable for NLP or LLM embeddings. |
| `raw.json` | JSON | **Immutable** Whisper output with segment metadata and confidence scores. |
| `corrected.json` | JSON | Standardized transcript with normalized terminology. |
| `corrections.json` | JSON | Auditable diff log recording every substitution, rejection, and flag. |

### Sample Markdown Output (`transcript.md`)

```markdown
# 001 Architecture Overview

- source: `001-architecture-overview.mp4`
- duration: 00:45:12
- model: mlx-community/whisper-large-v3-mlx
- language: vi
- segments: 340
- needs_review: 2

## Transcript

[00:01:15 - 00:01:22]
Chào mừng các bạn đến với phần tổng quan kiến trúc hệ thống phân tán.

[00:01:22 - 00:01:30]
Hôm nay chúng ta sẽ tìm hiểu về cách kết nối các dịch vụ CloudFront và API Gateway.

[00:01:30 - 00:01:38] <!-- needs_review -->
Phần này yêu cầu am hiểu rõ về chính sách bảo mật mạng và định tuyến.
```

---

## Testing

The project includes an exhaustive unit and integration test suite:

```bash
# Run unit tests (46 tests: paths, DB, VAD, rules, validation gates)
uv run pytest tests/unit -v

# Run integration tests (FFmpeg synthesis, format assertions, crash recovery)
uv run pytest tests/integration -v -m "not requires_model"

# Run complete test suite including local Whisper model loading
uv run pytest -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
