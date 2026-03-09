import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

LOGS = {
    "GPT-5": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fix_log.json"),
    "DeepSeek": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fix_log.json"),
    # "Claude": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
}


def analyze_log(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)

    total = len(log)
    fixed = [v for v in log.values() if v["status"] == "fixed"]
    unfixed = [v for v in log.values() if v["status"] == "unfixed"]
    llm_errors = [v for v in log.values() if v["status"] == "llm_error"]

    avg_attempts = sum(v["attempts"] for v in fixed) / len(fixed) if fixed else 0
    avg_time = sum(v["time_seconds"] for v in log.values()) / total if total else 0
    total_time = sum(v["time_seconds"] for v in log.values())

    return {
        "total": total,
        "fixed": len(fixed),
        "unfixed": len(unfixed),
        "llm_errors": len(llm_errors),
        "fix_rate": 100 * len(fixed) / total if total else 0,
        "avg_attempts_to_fix": round(avg_attempts, 2),
        "avg_time_per_file": round(avg_time, 2),
        "total_time": round(total_time, 2),
    }


if __name__ == "__main__":
    for model_name, log_path in LOGS.items():
        if not os.path.exists(log_path):
            print(f"{model_name}: log not found at {log_path}\n")
            continue
        stats = analyze_log(log_path)
        print(f"=== {model_name} ===")
        print(f"  Total files:          {stats['total']}")
        print(f"  Fixed:                {stats['fixed']}")
        print(f"  Unfixed:              {stats['unfixed']}")
        print(f"  LLM errors:           {stats['llm_errors']}")
        print(f"  Fix rate:             {stats['fix_rate']:.1f}%")
        print(f"  Avg attempts to fix:  {stats['avg_attempts_to_fix']}")
        print(f"  Avg time per file:    {stats['avg_time_per_file']}s")
        print(f"  Total time:           {stats['total_time']}s")
        print()
