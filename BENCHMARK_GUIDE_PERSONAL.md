# Benchmark Guide (Personal)

Tài liệu này là bản ghi chú cá nhân để hiểu nhanh benchmark trong Lab 16 (ReAct vs Reflexion), cách chạy, cách đọc report và cách tối ưu điểm autograde.

## 1) Benchmark trong lab này là gì?

Benchmark là quy trình chạy 2 agent trên cùng dataset câu hỏi:

- `react`: trả lời 1 lần.
- `reflexion`: có nhiều attempt, tự phản tư sau lần sai rồi thử lại.

Mục tiêu là so sánh:

- độ chính xác (`EM`),
- số lần thử (`attempts`),
- chi phí token (`token_estimate`),
- độ trễ (`latency_ms`).

Kết quả được gom vào report JSON/Markdown để autograde chấm.

## 2) Dữ liệu benchmark

- File nhỏ để smoke test: `data/hotpot_mini.json` (8 mẫu).
- File benchmark thực tế: `data/hotpot_100.json` (100 mẫu từ HotpotQA qua HuggingFace, đã convert về format `QAExample`).

Schema mỗi mẫu:

- `qid`
- `difficulty`
- `question`
- `gold_answer`
- `context` (list các chunk có `title`, `text`)

## 3) Các chế độ chạy

`run_benchmark.py` hỗ trợ:

- `--mode mock`: chạy bằng mock runtime (nhanh, dùng để kiểm tra pipeline).
- `--mode real`: chạy LLM thật qua OpenAI-compatible endpoint.

Trong `real` mode:

- Actor/Reflector dùng `default_model` (hoặc `DEFAULT_MODEL`).
- Evaluator dùng `judge_model` (hoặc `JUDGE_MODEL`) để làm LLM-as-a-judge.
- Endpoint lấy từ `DEFAULT_BASE_URL`.
- API key ưu tiên `default_api_key`, fallback `DEFAULT_API_KEY`.

## 4) Cách chạy chuẩn

## 4.1 Chuẩn bị dataset 100 mẫu

```bash
python scripts/prepare_dataset.py --out data/hotpot_100.json --limit 100
```

## 4.2 Chạy benchmark

Smoke test:

```bash
python run_benchmark.py --mode mock --dataset data/hotpot_mini.json --out-dir outputs/sample_run --reflexion-attempts 3
```

Run thật:

```bash
python run_benchmark.py --mode real --dataset data/hotpot_100.json --out-dir outputs/real_run --reflexion-attempts 3
```

## 4.3 Chạy autograde

```bash
python autograde.py --report-path outputs/real_run/report.json
```

## 5) Cách đọc report

Các file output chính:

- `outputs/.../react_runs.jsonl`
- `outputs/.../reflexion_runs.jsonl`
- `outputs/.../report.json`
- `outputs/.../report.md`

Các phần quan trọng trong `report.json`:

- `meta`: dataset, mode, số record.
- `summary`: EM/attempts/token/latency theo từng agent và delta.
- `failure_modes`: breakdown lỗi.
- `examples`: ví dụ per-record.
- `extensions`: các extension đã implement.
- `discussion`: phần nhận xét tổng kết.

## 6) Rubric autograde (điểm số)

`autograde.py` chấm theo 2 phần:

Core flow (80 điểm):

- Schema completeness (30): có đủ 6 key top-level.
- Experiment completeness (30):
  - có cả `react` và `reflexion`,
  - `num_records >= 100`,
  - `examples >= 20`.
- Analysis depth (20):
  - `failure_modes` đủ chiều sâu,
  - `discussion` đủ dài (>= 250 ký tự).

Bonus (20 điểm):

- Tính theo số extension recognized, tối đa 20.
- Một số extension được nhận: `structured_evaluator`, `reflection_memory`, `adaptive_max_attempts`, `memory_compression`, ...

## 7) Ý nghĩa các chỉ số benchmark

- `EM` (Exact Match sau normalize): càng cao càng tốt.
- `avg_attempts`: Reflexion thường cao hơn ReAct (đổi lại có thể tăng EM).
- `avg_token_estimate`: phản ánh cost.
- `avg_latency_ms`: phản ánh tốc độ.
- `delta_reflexion_minus_react`: giúp nhìn trade-off trực tiếp.

Diễn giải chuẩn:

- Reflexion tăng EM nhưng token/latency tăng là bình thường.
- Nếu EM không tăng mà chi phí tăng -> cần chỉnh prompt/reflector hoặc evaluator.

## 8) Checklist cá nhân trước khi nộp lab

- Chạy `mock` mode pass.
- Tạo đủ 100 mẫu trong `data/hotpot_100.json`.
- Chạy `real` mode thành công.
- `report.json` có đủ 6 key bắt buộc.
- `summary` có cả `react` + `reflexion`.
- `meta.num_records >= 100`.
- `examples` có ít nhất 20 item.
- `extensions` có ít nhất 2 mục recognized.
- `discussion` >= 250 ký tự.
- `autograde.py` đạt mức điểm mục tiêu.

## 9) Troubleshooting nhanh

- Lỗi `unknown provider for model ...`:
  - model trong `.env` không tồn tại trên gateway hiện tại.
  - đổi `default_model`/`judge_model` sang model ID hợp lệ của endpoint.

- Lỗi parse JSON evaluator/reflector:
  - đảm bảo vẫn dùng JSON mode (`response_format`),
  - prompt phải yêu cầu trả về JSON-only.

- Chạy lâu/rate limit:
  - giảm `reflexion_attempts`,
  - chạy nhỏ trước (`hotpot_mini.json`) để xác nhận pipeline.

---

Nếu cần, có thể tách file này thành bản ngắn để nộp (1 trang) và bản dài để học nội bộ.
