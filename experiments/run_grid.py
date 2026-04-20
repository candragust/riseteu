import json
import subprocess
from pathlib import Path


def load_plan(plan_path: Path):
    with plan_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_job(base_cfg: Path, job: dict, idx: int):
    name = job["name"]
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{idx:02d}_{name}_preds.csv"
    hist_csv = out_dir / f"{idx:02d}_{name}_history.csv"

    cmd = [
        "python",
        "bilstm_flf_experiment.py",
        "--config",
        str(base_cfg),
        "--window",
        str(job["window"]),
        "--units",
        str(job["units"]),
        "--activation",
        job["activation"],
        "--lr",
        str(job["lr"]),
        "--epochs",
        str(job["epochs"]),
        "--out",
        str(out_csv),
        "--history-out",
        str(hist_csv),
    ]
    print(f"[RUN {idx}] {name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    plan_path = Path(__file__).parent / "grid_plan.json"
    plan = load_plan(plan_path)
    base_cfg = Path(plan["base_config"])

    for idx, job in enumerate(plan["runs"], start=1):
        run_job(base_cfg, job, idx)


if __name__ == "__main__":
    main()

