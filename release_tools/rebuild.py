"""Rebuild from saved predictions in scratch space; never fit a learner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out == ROOT / "results" or out.exists():
        raise ValueError("Choose a new scratch directory, not the archived results directory")
    shutil.copytree(ROOT / "results", out)
    env = dict(os.environ, TABR1_RESULTS_ROOT=str(out), MPLBACKEND="Agg")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = "12"

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    scripts = ["build_manuscript_assets.py", "reusability/compute_reusability_statistics.py",
               "reusability/fold_aware_contrast.py", "reusability/build_reusability_figures.py"]
    for name in scripts:
        path = ROOT / "03_analysis_and_figures" / name
        code = ("import importlib.util,sys; from pathlib import Path; "
                "p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); "
                "s=importlib.util.spec_from_file_location('release_analysis',p); "
                "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
                + ("m.N_BOOT=10; " if args.smoke and "statistics" in name or args.smoke and "fold_aware" in name else "")
                + "m.main()")
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                       env=env, preexec_fn=limit, check=True)
    print(f"Rebuild complete: {out}; smoke={args.smoke}")


if __name__ == "__main__":
    main()
