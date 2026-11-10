"""Write the synthetic test files: `python -m tests.testcases_cli [folder] [--all]` (`just testcases`).

Without arguments it rewrites `tests/data/synthetic` (the stored cases) and removes files of cases that no longer
exist. `--all` also writes the long cases that are normally only built when a test needs them.
"""

import sys
from pathlib import Path

from tests.conftest import SYNTHETIC_DIR
from tests.testcases import CASES


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    everything = "--all" in sys.argv
    folder = Path(args[0]) if args else SYNTHETIC_DIR
    folder.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for case in CASES:
        if not case.stored and not everything:
            continue
        built = case.build()
        (folder / f"{case.id}.tdms").write_bytes(built.data)
        wanted.add(f"{case.id}.tdms")
        print(f"{case.id + '.tdms':44} {len(built.data):>9} bytes  {', '.join(case.tags)}")
    for old in folder.glob("*.tdms"):
        if old.name not in wanted:
            old.unlink()
            print(f"removed {old.name}")
    print(f"{len(wanted)} files in {folder}")


if __name__ == "__main__":
    main()
