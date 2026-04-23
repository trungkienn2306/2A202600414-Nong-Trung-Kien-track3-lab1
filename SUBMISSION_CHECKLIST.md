# Submission Checklist

## Files to commit
- `run_benchmark.py`
- `src/reflexion_lab/agents.py`
- `src/reflexion_lab/mock_runtime.py`
- `src/reflexion_lab/real_runtime.py`
- `src/reflexion_lab/reporting.py`
- `src/reflexion_lab/logging_utils.py`
- `outputs/final_real100/report.json`
- `outputs/final_real100/report.md`

## Files not to commit
- `.env`
- `outputs/**` logs, partial files, and run traces other than two report files above
- `__pycache__/` and `*.pyc`

## Quick check before commit
- Run `python autograde.py --report-path outputs/final_real100/report.json`
- Ensure `meta.num_records >= 100`
- Ensure both `react` and `reflexion` appear in summary
- Ensure `report.json` includes required top-level keys:
  - `meta`
  - `summary`
  - `failure_modes`
  - `examples`
  - `extensions`
  - `discussion`
