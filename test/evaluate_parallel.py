"""
Parallelized evaluation of the closed-loop executor across skills × env-ids × seeds.

Runs 9 workers simultaneously (3 skills × 3 env_ids). Each worker writes its own
log (.txt) and incremental results (.json). After all workers finish, the main
process prints the combined summary table.

Usage:
    python test/evaluate_parallel.py
    python test/evaluate_parallel.py --skills mpc ppo --seeds 5 --results-dir ./eval_results
    python test/evaluate_parallel.py --offline
"""
import argparse
import json
import multiprocessing
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "high_level_planner"))

from executor import run, _resolve_checkpoint  # noqa: E402

ENV_IDS = [
    "PushO-WallObstacles-v1",
    "PushO-Scattered",
    "PushO-TrappedDisk",
]
SKILLS = ["mpc", "ppo", "auto"]
N_SEEDS = 50


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _worker(args_dict: dict) -> dict:
    """Run all seeds for a single (skill, env_id) pair; writes its own log + JSON."""
    skill       = args_dict["skill"]
    env_id      = args_dict["env_id"]
    n_seeds     = args_dict["n_seeds"]
    max_replans = args_dict["max_replans"]
    offline     = args_dict["offline"]
    model       = args_dict["model"]
    reach_ckpt  = args_dict["reach_ckpt"]
    pick_ckpt   = args_dict["pick_ckpt"]
    place_ckpt  = args_dict["place_ckpt"]
    push_o_ckpt = args_dict["push_o_ckpt"]
    reach_device = args_dict["reach_device"]
    results_dir = Path(args_dict["results_dir"])
    timestamp   = args_dict["timestamp"]

    safe_env = env_id.replace("/", "_").replace("-", "_")
    log_path  = results_dir / f"eval_{timestamp}_{skill}_{safe_env}.txt"
    json_path = results_dir / f"results_{skill}_{safe_env}.json"

    success_list = []
    elapsed_list = []
    replans_list = []
    fails_list   = []

    # Redirect this process's stdout to the per-worker log file (tee to original fd).
    orig_stdout = sys.stdout
    log_f = open(log_path, "w", buffering=1)

    class _Tee:
        def write(self, data):
            orig_stdout.write(data)
            log_f.write(data)
        def flush(self):
            orig_stdout.flush()
            log_f.flush()
        def fileno(self):
            return orig_stdout.fileno()

    sys.stdout = _Tee()

    try:
        print(f"[{skill}/{env_id}] started  seeds=0..{n_seeds-1}")
        print(f"[{skill}/{env_id}] started: {datetime.now().isoformat()}")

        for seed in range(n_seeds):
            print(f"\n[{skill}/{env_id}] seed={seed}/{n_seeds-1}")
            t0 = time.perf_counter()
            try:
                ret = run(
                    seed=seed,
                    max_replans=max_replans,
                    offline=offline,
                    model=model,
                    skill=skill,
                    env_id=env_id,
                    checkpoint=reach_ckpt        if skill in ("ppo", "auto") else None,
                    pick_checkpoint=pick_ckpt    if skill in ("ppo", "auto") else None,
                    place_checkpoint=place_ckpt  if skill in ("ppo", "auto") else None,
                    push_o_checkpoint=push_o_ckpt if skill in ("ppo", "auto") else None,
                    reach_device=reach_device,
                )
                success      = ret["success"]
                n_replans    = ret["n_replans"]
                skill_fail_c = ret["skill_fail_counts"]
            except Exception:
                traceback.print_exc()
                success, n_replans, skill_fail_c = False, max_replans, {}

            elapsed = time.perf_counter() - t0
            success_list.append(bool(success))
            elapsed_list.append(round(elapsed, 2))
            replans_list.append(n_replans)
            fails_list.append(skill_fail_c)

            print(
                f"  → {'SUCCESS' if success else 'FAIL'}"
                f"  |  time={_fmt_time(elapsed)}"
                f"  |  replans={n_replans}"
                f"  |  skill_fails={skill_fail_c}"
            )

            with open(json_path, "w") as jf:
                json.dump({
                    "success":          success_list,
                    "elapsed_s":        elapsed_list,
                    "n_replans":        replans_list,
                    "skill_fail_counts": fails_list,
                }, jf, indent=2)

        print(f"\n[{skill}/{env_id}] finished: {datetime.now().isoformat()}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()

    return {
        "skill":            skill,
        "env_id":           env_id,
        "success":          success_list,
        "elapsed_s":        elapsed_list,
        "n_replans":        replans_list,
        "skill_fail_counts": fails_list,
        "log_path":         str(log_path),
        "json_path":        str(json_path),
    }


# ── Summary table (identical logic to evaluate.py) ────────────────────────────

def _print_table(results, times, replans, skill_fails):
    skills = sorted(results)
    envs   = ENV_IDS
    sep    = "=" * 110

    print(f"\n{sep}")
    print(f"{'METRIC':<22}" + "".join(f"{e:<30}" for e in envs))
    print("-" * 110)

    for skill in skills:
        print(f"\n  skill = {skill}")
        for label, getter in [
            ("success rate",   lambda sk, ev: _success_cell(results, sk, ev)),
            ("avg time",       lambda sk, ev: _time_cell(times, sk, ev)),
            ("avg replans",    lambda sk, ev: _replans_cell(replans, sk, ev)),
            ("top fail skill", lambda sk, ev: _fail_cell(skill_fails, sk, ev)),
        ]:
            row = f"  {label:<20}" + "".join(f"{getter(skill, ev):<30}" for ev in envs)
            print(row)

    print(sep)


def _success_cell(results, skill, env_id):
    outcomes = results[skill].get(env_id, [])
    if not outcomes:
        return "—"
    s, n = sum(outcomes), len(outcomes)
    return f"{s}/{n} ({100*s/n:.0f}%)"


def _time_cell(times, skill, env_id):
    ts = times[skill].get(env_id, [])
    if not ts:
        return "—"
    avg = sum(ts) / len(ts)
    return f"avg {_fmt_time(avg)} [{_fmt_time(min(ts))}–{_fmt_time(max(ts))}]"


def _replans_cell(replans, skill, env_id):
    rs = replans[skill].get(env_id, [])
    if not rs:
        return "—"
    return f"avg {sum(rs)/len(rs):.1f} [{min(rs)}–{max(rs)}]"


def _fail_cell(skill_fails, skill, env_id):
    agg: dict[str, int] = defaultdict(int)
    for run_fails in skill_fails[skill].get(env_id, []):
        for k, v in run_fails.items():
            agg[k] += v
    if not agg:
        return "none"
    top = max(agg, key=agg.get)
    return f"{top} ({agg[top]}x)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills",            nargs="+", default=SKILLS,   choices=SKILLS)
    ap.add_argument("--env-ids",           nargs="+", default=ENV_IDS,  dest="env_ids", choices=ENV_IDS)
    ap.add_argument("--seeds",             type=int,  default=N_SEEDS,  help="Seeds per (skill, env_id) worker")
    ap.add_argument("--max-replans",       type=int,  default=5,        dest="max_replans")
    ap.add_argument("--offline",           action="store_true")
    ap.add_argument("--model",             default="gemini-2.5-flash")
    ap.add_argument("--reach-checkpoint",  default="Reach",             dest="reach_checkpoint")
    ap.add_argument("--pick-checkpoint",   default="PickSkill",         dest="pick_checkpoint")
    ap.add_argument("--place-checkpoint",  default="PlaceSkill",        dest="place_checkpoint")
    ap.add_argument("--push-o-checkpoint", default="PushO",             dest="push_o_checkpoint")
    ap.add_argument("--reach-device",      default=None,                dest="reach_device")
    ap.add_argument("--results-dir",       default="./eval_results",    dest="results_dir")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_log = results_dir / f"eval_{timestamp}_summary.txt"

    reach_ckpt  = _resolve_checkpoint(args.reach_checkpoint)
    pick_ckpt   = _resolve_checkpoint(args.pick_checkpoint)
    place_ckpt  = _resolve_checkpoint(args.place_checkpoint)
    push_o_ckpt = _resolve_checkpoint(args.push_o_checkpoint)

    worker_args = [
        {
            "skill":       skill,
            "env_id":      env_id,
            "n_seeds":     args.seeds,
            "max_replans": args.max_replans,
            "offline":     args.offline,
            "model":       args.model,
            "reach_ckpt":  reach_ckpt,
            "pick_ckpt":   pick_ckpt,
            "place_ckpt":  place_ckpt,
            "push_o_ckpt": push_o_ckpt,
            "reach_device": args.reach_device,
            "results_dir": str(results_dir),
            "timestamp":   timestamp,
        }
        for skill  in args.skills
        for env_id in args.env_ids
    ]

    n_workers = len(worker_args)
    print(f"Launching {n_workers} parallel workers ({len(args.skills)} skills × {len(args.env_ids)} envs), {args.seeds} seeds each.")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Per-worker logs: {results_dir}/eval_{timestamp}_<skill>_<env>.txt\n")

    t_total = time.perf_counter()
    with multiprocessing.Pool(processes=n_workers) as pool:
        worker_results = pool.map(_worker, worker_args)

    elapsed_total = time.perf_counter() - t_total
    print(f"\nAll workers finished in {_fmt_time(elapsed_total)}.")

    # Aggregate into summary dicts
    results      = {sk: {ev: [] for ev in ENV_IDS} for sk in SKILLS}
    times_store  = {sk: {ev: [] for ev in ENV_IDS} for sk in SKILLS}
    replans_store= {sk: {ev: [] for ev in ENV_IDS} for sk in SKILLS}
    fails_store  = {sk: {ev: [] for ev in ENV_IDS} for sk in SKILLS}

    for wr in worker_results:
        sk, ev = wr["skill"], wr["env_id"]
        results[sk][ev]       = wr["success"]
        times_store[sk][ev]   = wr["elapsed_s"]
        replans_store[sk][ev] = wr["n_replans"]
        fails_store[sk][ev]   = wr["skill_fail_counts"]
        print(f"  [{sk}/{ev}]  log → {wr['log_path']}")
        print(f"  [{sk}/{ev}]  json → {wr['json_path']}")

    print(f"\nFinished: {datetime.now().isoformat()}")
    _print_table(results, times_store, replans_store, fails_store)

    # Write summary log
    with open(summary_log, "w") as sf:
        sf.write(f"Parallel evaluation summary\n")
        sf.write(f"Started: {datetime.now().isoformat()}\n")
        sf.write(f"Skills:  {args.skills}\n")
        sf.write(f"Env IDs: {args.env_ids}\n")
        sf.write(f"Seeds:   {args.seeds}\n\n")
        for wr in worker_results:
            sk, ev = wr["skill"], wr["env_id"]
            outcomes = wr["success"]
            s, n = sum(outcomes), len(outcomes)
            sf.write(f"  {sk:<6} {ev:<30} {s}/{n} ({100*s/n:.0f}%)\n")
    print(f"\nSummary log: {summary_log}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
