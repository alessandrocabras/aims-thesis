"""Run the preserved figure renderer in an isolated, ignored directory."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "thesis_figures"
    target = ROOT / "reproduced/thesis_figures"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("make_thesis_figures.py", "verify_figures.py"):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / "source_data", target / "source_data", dirs_exist_ok=True)
    subprocess.run([sys.executable, str(target / "make_thesis_figures.py")], check=True)
    print("Regenerated figures: reproduced/thesis_figures/. Published originals unchanged.")


if __name__ == "__main__":
    main()
