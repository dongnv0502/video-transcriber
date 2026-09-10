# AWS Video Transcriber — Technical Specification

## 1. Mục tiêu

Xây dựng một CLI chạy **100% local trên MacBook Apple Silicon**, dùng để tự động bóc tách transcript từ khoảng 200 video khóa học AWS tiếng Việt ở định dạng MP4.

Ưu tiên theo thứ tự:

1. Độ chính xác transcript.
2. Giữ timestamp để có thể truy ngược transcript về video.
3. Xử lý batch ổn định, có resume/checkpoint.
4. Không upload audio/video/transcript lên cloud.
5. Có lớp hiệu chỉnh AWS terminology nhưng **không làm thay đổi nội dung bài giảng ngoài các lỗi ASR có cơ sở**.
6. Có output phù hợp cho AI coding/learning agents.

## 2. Phạm vi

### In scope

- Scan thư mục MP4.
- Extract audio bằng FFmpeg.
- Chuẩn hóa audio về format phù hợp ASR.
- Voice Activity Detection (VAD).
- Transcription bằng Whisper `large-v3` chạy local trên Apple Silicon.
- Lưu timestamp theo segment.
- Phát hiện segment đáng nghi.
- Hiệu chỉnh AWS terminology bằng dictionary + local LLM.
- Lưu raw transcript bất biến.
- Xuất transcript corrected dưới dạng JSON/Markdown/TXT/SRT.
- SQLite để quản lý trạng thái batch.
- Resume sau lỗi/crash/restart.
- Logging và báo cáo tiến độ.
- Benchmark trên 3 video mẫu trước khi chạy toàn bộ dataset.

### Out of scope ở phase 1

- Web UI.
- Server/API.
- Cloud inference.
- Fine-tuning Whisper.
- RAG/vector database.
- Tự động tạo summary/flashcard/quiz trong cùng project.

## 3. Nguyên tắc kiến trúc

### 3.1 Local-only

Toàn bộ pipeline phải chạy local. Không gọi OpenAI, Anthropic, Google, AWS Bedrock hoặc bất kỳ inference/transcription API cloud nào.

Network chỉ được phép khi người dùng chủ động tải model/dependency.

### 3.2 Raw transcript là source of truth

Không bao giờ overwrite raw transcript bằng bản corrected.

Luôn lưu tối thiểu:

```text
raw.json
corrected.json
transcript.md
transcript.srt
```

### 3.3 Không rewrite nội dung

Correction layer chỉ được phép:

- sửa lỗi ASR rõ ràng;
- chuẩn hóa AWS technical terminology khi có đủ bằng chứng;
- sửa lỗi nhận dạng tên service, acronym, command hoặc technical term.

Không được:

- summarize;
- paraphrase;
- rút gọn câu;
- thay đổi phong cách người giảng;
- thêm kiến thức từ bên ngoài;
- tự suy diễn câu bị thiếu.

Nếu không chắc chắn: giữ nguyên raw transcript và đánh dấu `needs_review`.

## 4. Kiến trúc pipeline

```text
MP4
 │
 ▼
Scanner
 │
 ▼
FFmpeg audio extraction
 │
 ▼
Audio preprocessing
 │
 │  mono / 16 kHz / PCM
 ▼
VAD / speech segmentation
 │
 ▼
Whisper large-v3 (local, Apple Silicon)
 │
 ▼
Raw transcript + timestamps + metadata
 │
 ├───────────────┐
 ▼               ▼
AWS glossary   Suspicious detector
 │               │
 └───────┬───────┘
         ▼
Local LLM correction
         │
         ▼
Validation / diff
         │
    ┌────┴────┐
    │         │
    ▼         ▼
accepted   needs_review
    │
    ▼
Corrected transcript
    │
    ├── JSON
    ├── Markdown
    ├── TXT
    └── SRT
```

## 5. Technology choices

### 5.1 Language

Python 3.11+.

Lý do:

- hệ sinh thái ASR tốt;
- dễ tích hợp FFmpeg, SQLite và local LLM;
- phù hợp CLI/batch processing.

### 5.2 Audio

FFmpeg.

Target format:

```text
PCM WAV
mono
16 kHz
```

Không cần lưu audio đã extract nếu có thể tái tạo; tuy nhiên pipeline nên hỗ trợ cache audio để tránh extract lại khi rerun.

### 5.3 ASR

Baseline bắt buộc benchmark:

```text
Whisper large-v3
```

Ưu tiên backend tối ưu Apple Silicon, trước mắt đánh giá MLX Whisper.

Có thể benchmark thêm `faster-whisper` hoặc `whisper.cpp` nếu backend MLX không đạt yêu cầu về tốc độ/ổn định.

Không khóa cứng backend trước benchmark.

### 5.4 VAD

Dùng một VAD local phù hợp với Python pipeline.

Mục tiêu:

- bỏ khoảng silence dài;
- hỗ trợ speech-aware segmentation;
- không cắt giữa câu nếu có thể tránh.

### 5.5 Local LLM

Dùng local inference, ví dụ Ollama hoặc llama.cpp.

Target model initial benchmark: khoảng 7B–14B instruct model phù hợp 24 GB unified memory.

Model phải được cấu hình local-only.

### 5.6 Storage

SQLite.

Không cần PostgreSQL/Redis cho phase 1.

## 6. Data model

### 6.1 Video record

```sql
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    status TEXT NOT NULL,
    model TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Status tối thiểu:

```text
PENDING
EXTRACTING
TRANSCRIBING
CORRECTING
COMPLETED
FAILED
NEEDS_REVIEW
```

### 6.2 Segment JSON

```json
{
  "start": 121.42,
  "end": 127.31,
  "text": "IAM Role cho phép EC2 truy cập S3",
  "confidence": 0.91,
  "corrected": false,
  "needs_review": false
}
```

Confidence phụ thuộc backend. Nếu backend không cung cấp confidence đáng tin cậy thì không tự tạo confidence giả.

### 6.3 Video transcript JSON

```json
{
  "schema_version": "1.0",
  "video": {
    "filename": "002-iam.mp4",
    "duration_seconds": 1234.5
  },
  "transcription": {
    "model": "whisper-large-v3",
    "language": "vi"
  },
  "segments": []
}
```

## 7. AWS terminology correction

Tạo glossary versioned:

```text
glossary/
├── aws.yml
└── corrections.yml
```

Ví dụ:

```yaml
terms:
  - IAM
  - IAM Role
  - IAM Policy
  - EC2
  - S3
  - VPC
  - EBS
  - EFS
  - ALB
  - NLB
  - CloudFront
  - CloudWatch
  - CloudFormation
  - Route 53
  - Lambda
  - API Gateway
  - ECS
  - EKS
  - Fargate
```

Correction mapping chỉ là candidate, không được blind string replacement.

Ví dụ candidate:

```yaml
corrections:
  - from: "I am role"
    to: "IAM Role"
```

Rule engine phải xét context để tránh false positive.

## 8. Suspicious detection

Mục tiêu là tìm segment có khả năng ASR sai.

Có thể dựa trên:

- confidence thấp nếu backend cung cấp;
- token/word bất thường;
- từ ngữ gần với AWS glossary nhưng không khớp;
- acronym/service name sai;
- output chứa pattern khả nghi;
- segment quá ngắn/quá bất thường.

Không tự động coi mọi suspicious segment là sai.

## 9. Local LLM correction contract

Prompt phải yêu cầu model trả về structured output.

Ví dụ schema:

```json
{
  "changed": true,
  "original": "Trong bài này chúng ta sẽ tìm hiểu về I am Role",
  "corrected": "Trong bài này chúng ta sẽ tìm hiểu về IAM Role",
  "reason": "Likely AWS terminology ASR error",
  "confidence": 0.98,
  "needs_review": false
}
```

Nếu model không đủ chắc chắn:

```json
{
  "changed": false,
  "corrected": null,
  "reason": "uncertain",
  "needs_review": true
}
```

LLM không được thêm thông tin mới.

## 10. Chunking

Không chunk audio một cách cứng nhắc theo mỗi 30 giây nếu VAD/speech boundary cho phép tốt hơn.

Ưu tiên:

```text
speech boundary
    ↓
context-aware segment
    ↓
Whisper
```

Đối với correction LLM, chunk theo lượng text/token hợp lý, giữ nguyên thứ tự và timestamp.

Không gửi toàn bộ transcript của video dài vào một prompt duy nhất.

## 11. Batch processing

Phải hỗ trợ:

```bash
transcriber scan ./videos
transcriber transcribe
transcriber correct
transcriber export
transcriber status
transcriber resume
```

Có thể thiết kế lệnh gộp:

```bash
transcriber process ./videos
```

Nhưng backend pipeline phải tách stage để có thể rerun correction mà không transcribe lại.

## 12. Checkpoint / resume

Nếu process crash ở video thứ 137 thì lần chạy tiếp theo chỉ xử lý pending/failed cần retry.

Ví dụ:

```text
200 total
137 completed
1 failed
62 pending
```

Không xử lý lại transcript đã completed nếu artifact hợp lệ.

Phải có cơ chế atomic write:

```text
*.tmp
   ↓
fsync/close
   ↓
rename
   ↓
final artifact
```

để tránh file transcript hỏng khi process bị kill giữa chừng.

## 13. Concurrency

Mặc định:

```text
ASR workers = 1
LLM workers = 1
```

Sau benchmark mới cho phép tăng lên 2 hoặc hơn nếu có lợi thực tế.

Không được tự động spawn quá nhiều worker chỉ dựa trên CPU core count.

## 14. CLI observability

Khi chạy batch cần hiển thị tối thiểu:

```text
[037/200] AWS IAM
progress: 82%
segments: 384
suspicious: 7
status: transcribing
```

Cuối mỗi video:

```text
completed
output: output/037/
duration: 46m12s
processing: 18m31s
segments: 384
needs_review: 7
```

## 15. Logging

Log vào:

```text
logs/app.log
```

Mỗi error cần có:

- video path;
- stage;
- exception;
- timestamp;
- retry information.

Không log nội dung transcript đầy đủ ở INFO level.

## 16. Output structure

```text
output/
├── 001-introduction/
│   ├── raw.json
│   ├── corrected.json
│   ├── transcript.md
│   ├── transcript.txt
│   └── transcript.srt
└── ...
```

Markdown nên dễ đọc cho AI Agent:

```markdown
# AWS IAM

## Transcript

[00:02:12 - 00:02:18]
IAM Role cho phép EC2...
```

## 17. Benchmark bắt buộc trước batch 200 video

Không chạy toàn bộ dataset ngay.

Chọn 3 video đại diện:

1. Bài giảng tiếng Việt bình thường.
2. Bài có nhiều AWS terminology.
3. Bài có CLI/code/demo.

Đo tối thiểu:

```text
- processing time
- real-time factor
- RAM usage
- GPU/Metal utilization nếu đo được
- ASR errors
- AWS terminology errors
- CLI/code recognition
- timestamp quality
- correction false positives
```

### Manual accuracy benchmark

Tạo ground-truth thủ công cho một mẫu câu/đoạn video.

Không chỉ dùng WER.

Phải đặc biệt đánh giá:

```text
EC2
IAM
IAM Role
S3
VPC
CloudFormation
CLI commands
numbers
URLs
technical English terms
```

## 18. Acceptance criteria

### Functional

- [ ] Scan được toàn bộ MP4 trong thư mục.
- [ ] Không cần cloud API để transcribe/correct.
- [ ] Một video có thể transcribe từ đầu đến cuối.
- [ ] Có timestamp cho segment.
- [ ] Có raw + corrected transcript tách biệt.
- [ ] Có SRT.
- [ ] Có Markdown phù hợp AI Agent.
- [ ] Có resume.
- [ ] Có retry cho failed video.
- [ ] Có SQLite status.
- [ ] Có logging.

### Quality

- [ ] Benchmark 3 video trước khi batch 200 video.
- [ ] Không overwrite raw transcript.
- [ ] Không tự ý paraphrase transcript.
- [ ] Correction output có structured diff.
- [ ] Uncertain correction được đánh dấu `needs_review`.
- [ ] AWS terminology được benchmark riêng.

### Privacy

- [ ] Không gọi cloud inference API.
- [ ] Không upload MP4/audio/transcript.
- [ ] Có documentation về network requirement của runtime.

## 19. Testing

### Unit tests

- filename/path handling;
- glossary matching;
- correction rules;
- JSON schema validation;
- status transitions;
- resume behavior;
- atomic output write.

### Integration tests

- FFmpeg extraction;
- one short audio → transcript;
- transcript → local correction;
- transcript → SRT/Markdown export;
- failed stage → resume.

### Regression test

Lưu một audio sample nhỏ + expected transcript fixture để kiểm tra backend/model/configuration thay đổi có làm quality regression.

## 20. Error handling

Nếu FFmpeg fail:

```text
EXTRACTING → FAILED
```

Nếu Whisper fail:

```text
TRANSCRIBING → FAILED
```

Nếu correction fail:

```text
CORRECTING → FAILED
```

Raw transcript đã tạo thành công phải được giữ lại để không phải transcribe lại chỉ vì correction fail.

## 21. Configuration

Dùng một config file hoặc environment variables cho:

```text
VIDEO_DIR
OUTPUT_DIR
MODEL_NAME
MODEL_BACKEND
LANGUAGE
VAD_ENABLED
LLM_ENABLED
LLM_MODEL
ASR_WORKERS
LLM_WORKERS
```

Không hard-code path của máy người dùng.

## 22. Recommended implementation order

### Phase 1

- CLI skeleton.
- Scanner.
- SQLite.
- FFmpeg extraction.
- Whisper large-v3 transcription.
- raw.json.

### Phase 2

- VAD.
- timestamp improvements.
- SRT/Markdown/TXT export.
- resume/checkpoint.

### Phase 3

- AWS glossary.
- suspicious detection.
- local LLM correction.
- correction diff.

### Phase 4

- benchmark tooling.
- quality report.
- tuning worker count/model/backend.

### Phase 5

- batch run 200 videos.

## 23. Coding-agent instructions

Agent phải:

1. Trước tiên kiểm tra môi trường local: macOS, Apple Silicon, Python, FFmpeg, available memory.
2. Không tự ý chọn model/backend chỉ dựa trên lý thuyết; tạo benchmark nhỏ để kiểm chứng.
3. Không implement toàn bộ pipeline một lần nếu chưa có một vertical slice chạy được.
4. Mỗi stage phải có input/output rõ ràng.
5. Ưu tiên code đơn giản, dễ debug, dễ rerun.
6. Không overwrite raw artifacts.
7. Không sử dụng cloud APIs.
8. Khi một correction không chắc chắn, giữ nguyên raw và đánh dấu review.
9. Sau mỗi implementation step phải chạy test tương ứng.
10. Không tuyên bố accuracy hoặc performance mà chưa benchmark trên máy thực tế.

## 24. Desired end state

Sau khi hoàn thành, chạy:

```bash
transcriber process ./videos
```

sẽ tạo được dataset:

```text
output/
├── 001/
│   ├── raw.json
│   ├── corrected.json
│   ├── transcript.md
│   ├── transcript.txt
│   └── transcript.srt
├── 002/
└── ...
```

Dataset này sẽ là input cho các AI learning agents ở bước tiếp theo để thực hiện:

```text
summary
flashcards
quiz
study notes
AWS concept extraction
interview questions
Q&A
```

Những chức năng trên **không thuộc transcript pipeline phase 1**; transcript pipeline chỉ cần cung cấp source dữ liệu chất lượng cao, có timestamp và có thể audit.
