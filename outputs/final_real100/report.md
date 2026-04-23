# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.59 | 0.83 | 0.24 |
| Avg attempts | 1 | 1.6 | 0.6 |
| Avg token estimate | 1343.37 | 2880.43 | 1537.06 |
| Avg latency (ms) | 17755.37 | 27515.71 | 9760.34 |

## Failure modes
```json
{
  "react": {
    "wrong_final_answer": 41,
    "none": 59
  },
  "reflexion": {
    "none": 83,
    "wrong_final_answer": 17
  },
  "overall": {
    "wrong_final_answer": 58,
    "none": 142
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
