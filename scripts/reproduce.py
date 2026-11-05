"""
Run the report's experiments, tables and figures from configs/report/reproduce.yml.

Each training stage is a grid of cells, one command per cell. A cell whose metrics.json
already exists is skipped, so an interrupted run resumes where it stopped and a dry run
on a machine holding the original results reports every cell as done.

Output of every command is echoed and appended to outputs/reproduce/<stage>.log.

Usage (from Model/):
    python -m scripts.reproduce --dry-run
    python -m scripts.reproduce --stage node21 vindr
    python -m scripts.reproduce --stage node21 --smoke 1     # one 1-epoch cell, own run group
    python -m scripts.reproduce                              # everything, in order
"""

import argparse
import glob
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.vindr import VINDR_ROOT                       # noqa: E402

CONFIG = ROOT / "configs" / "report" / "reproduce.yml"
LOG_DIR = ROOT / "outputs" / "reproduce"

STAGE_ORDER = ["prepare", "node21", "vindr", "score", "tables", "figures"]
RUN_PLACEHOLDER = re.compile(r"\{run:([^}]*)\}")


def field(metrics, dotted):
    """Read a possibly nested metrics.json field by a dotted key, None when absent."""
    value = metrics

    for key in dotted.split("."):
        if not isinstance(value, dict): return None
        value = value.get(key)

    return value


def index_runs():
    """Every metrics.json under outputs/, parsed once, paired with its directory."""
    runs = []

    for path in (ROOT / "outputs").rglob("metrics.json"):
        try:
            runs.append((path.parent, json.load(open(path))))
        except (json.JSONDecodeError, OSError):
            continue

    return runs


def find_run(runs, wanted):
    """First run directory whose fields match every (dotted key -> string) pair."""
    for directory, metrics in runs:
        if all(str(field(metrics, key)) == value for key, value in wanted.items()):
            return directory

    return None


def expand(grids):
    """Cartesian product of each grid entry's lists, in file order."""
    for grid in grids:
        keys = list(grid)

        for values in itertools.product(*(grid[key] for key in keys)):
            yield dict(zip(keys, values))


def to_argv(command, values, runs):
    """
    Fill placeholders and split into an argument list.

    `python` as the first word becomes this interpreter, so commands run in the same
    environment as the runner. No shell is involved, which keeps quoting identical on
    Windows and POSIX.
    """
    def resolve(match):
        wanted = dict(pair.split("=", 1) for pair in match.group(1).split(","))
        directory = find_run(runs, wanted)

        if directory is None:
            raise SystemExit("no run matches {}".format(match.group(0)))

        # Quoted forward-slash form: the path may hold spaces, and shlex would read a
        # Windows backslash as an escape
        return '"{}"'.format(directory.as_posix())

    command = RUN_PLACEHOLDER.sub(resolve, command).format(**values)
    argv = shlex.split(command, posix=True)

    if argv[0] == "python":
        argv[0] = sys.executable

    return argv


def execute(argv, log_path, environment):
    """Run one command from Model/, echoing and logging its output. Returns exit code."""
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ {}\n".format(" ".join(argv)))

        process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace")

        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)

        return process.wait()


def run_stage(name, stage, paths, arguments, runs, environment):
    """Expand and execute one stage. Returns the number of failed commands."""
    print("\n=== {} ===".format(name))

    log_path = LOG_DIR / "{}.log".format(name)
    failures = 0

    # Training stages: one command per grid cell, skipping finished cells
    if "grids" in stage:
        cells = list(expand(stage["grids"]))

        if arguments.smoke:
            cells = cells[:arguments.smoke]

        for number, cell in enumerate(cells, start=1):
            values = dict(paths, **cell)

            # Named after the stage, never the real group: the figure scripts match run
            # groups by substring, so smoke_v15_... would be pulled into the report figures
            if arguments.smoke:
                values["run_group"] = "smoke_" + name

            wanted = {key: str(value).format(**values) for key, value in stage["done_if"].items()}
            finished = find_run(runs, wanted)
            label = "[{}/{}] {}".format(number, len(cells),
                                        " ".join("{}={}".format(k, v) for k, v in cell.items()))

            if finished and not arguments.force:
                print("  done  {}  ({})".format(label, finished.relative_to(ROOT).as_posix()))
                continue

            argv = to_argv(stage["command"], values, runs)

            if arguments.smoke:
                argv += shlex.split(arguments.smoke_args)

            print("  todo  {}".format(label))

            if arguments.dry_run:
                print("        {}".format(" ".join(argv[1:])))
                continue

            failures += execute(argv, log_path, environment) != 0
            runs[:] = index_runs()

        return failures

    # Command stages: run each in turn, then check expectations and copy outputs
    for command in stage.get("commands", []):
        # A dry run shows the template: {run:...} has nothing to resolve on a fresh machine
        print("  $ {}".format(command.replace("{figures}", paths["figures"])))

        if not arguments.dry_run:
            failures += execute(to_argv(command, paths, runs), log_path, environment) != 0

    if arguments.dry_run:
        return failures

    for expectation in stage.get("expect", []):
        found = len(glob.glob(expectation["glob"].format(**paths)))
        ok = found == expectation["count"]
        failures += not ok

        print("  {}  {} files match {} (expected {})".format(
            "ok  " if ok else "FAIL", found, expectation["glob"].format(**paths),
            expectation["count"]))

    for pattern, target in stage.get("copy", {}).items():
        matches = sorted(glob.glob(str(ROOT / paths["figures"] / pattern)))

        if len(matches) != 1:
            print("  FAIL  {} matched {} files".format(pattern, len(matches)))
            failures += 1
            continue

        destination = ROOT / paths["figures"] / "report" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], destination)
        print("  copy  {} -> report/{}".format(Path(matches[0]).name, target))

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--stage", nargs="*", default=STAGE_ORDER, choices=STAGE_ORDER)
    parser.add_argument("--dry-run", action="store_true", help="list commands, run nothing")
    parser.add_argument("--force", action="store_true", help="retrain cells already on disk")
    parser.add_argument("--smoke", type=int, default=0,
                        help="run only the first N cells of each training stage, for 1 epoch, "
                             "under a smoke_ run group")
    arguments = parser.parse_args()

    config = yaml.safe_load(open(arguments.config))

    paths = dict(config["paths"], vindr_root=VINDR_ROOT.as_posix())
    arguments.smoke_args = config.get("smoke_args", "--epochs 1")

    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / paths["figures"]).mkdir(parents=True, exist_ok=True)

    runs = index_runs()
    failures = 0

    for name in [stage for stage in STAGE_ORDER if stage in arguments.stage]:
        failures += run_stage(name, config["stages"][name], paths, arguments, runs, environment)

    print("\n{} failed command(s). Logs in {}".format(failures, LOG_DIR.relative_to(ROOT).as_posix()))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
