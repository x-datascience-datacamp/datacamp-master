"""Check the course notebooks against the conventions of CONTRIBUTING.md.

Usage: python tools/check_notebooks.py
Prints one line per problem and exits with 1 if there is any.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/x-datascience-datacamp/datacamp-master"
BADGE = (
    "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/x-datascience-datacamp/datacamp-master/blob/main/{path})"
)
SETUP = '''# Colab: clone the course repository and run from this notebook's folder
import os, subprocess, sys

if "google.colab" in sys.modules:
    repo = "/content/datacamp-master"
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1", "{repo}", repo], check=True)
    os.chdir(f"{{repo}}/{folder}")'''

NAME = re.compile(r"\d\d[a-z]?-[A-Za-z0-9_-]+\.ipynb")
SOLUTION_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*-\d+_[A-Za-z0-9_-]+\.py")
MARKER = re.compile(r"(<b>|\*\*)\s*(exercise|questions?)\b[^<*]*(</b>|\*\*)\s*:?", re.I)
BOX = re.compile(r"<b>(EXERCISE|QUESTION)</b>:")
SOLUTION = re.compile(r"^Solution:.*$", re.M)
SOLUTION_LINE = re.compile(r"Solution: `solutions/([^`]+)`")
COLAB = re.compile(r"\[!\[Open in Colab\]\([^)]*\)\]\([^)]*\)")
PIP = re.compile(r"^\s*[!%]\s*pip\b", re.M)

paths = [
    p for p in subprocess.run(["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=True).stdout.split()
    if not p.startswith("old_materials/")
]
errors = []
referenced = set()
for path in paths:
    cells = json.loads(Path(path).read_text())["cells"]
    sources = ["".join(c["source"]) for c in cells]
    folder = str(Path(path).parent)
    error = lambda i, msg: errors.append(f"{path}: cell {i}: {msg}")

    if not NAME.fullmatch(Path(path).name):
        error(0, "name the notebook NN-topic.ipynb")
    lines = sources[0].split("\n")
    if cells[0]["cell_type"] != "markdown" or not lines[0].startswith("# ") or len(lines) < 3 \
            or lines[2] != BADGE.format(path=path):
        error(0, "start with '# Title', a blank line and the Colab badge of this notebook")
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    if sources[first_code] != SETUP.format(repo=REPO, folder=folder):
        error(first_code, "the first code cell must be the Colab setup cell")

    for i, (cell, text) in enumerate(zip(cells, sources)):
        if cell["cell_type"] == "code":
            if PIP.search(text):
                error(i, "comment installs out: '# !pip install -U <package>'")
            if re.search(r"%load\s", text):
                error(i, "no %load: name the solution in the exercise box")
            continue
        if any(b != BADGE.format(path=path) for b in COLAB.findall(text)):
            error(i, "use the Colab badge of this notebook")
        markers = [m.group(0) for m in MARKER.finditer(text)]
        if any(not BOX.fullmatch(m) for m in markers) or (markers and 'class="alert alert-success"' not in text):
            error(i, "write boxes as <div class=\"alert alert-success\"> with <b>EXERCISE</b>: or <b>QUESTION</b>:")
        solutions = SOLUTION.findall(text)
        if solutions and "<b>EXERCISE</b>:" not in text:
            error(i, "only an EXERCISE box names a solution")
        if len(solutions) > 1 or any(not SOLUTION_LINE.fullmatch(s) for s in solutions):
            error(i, "name at most one solution, on its own line: Solution: `solutions/<file>.py`")
        if "<b>EXERCISE</b>:" in text and (i + 1 == len(cells) or cells[i + 1]["cell_type"] != "code"):
            error(i, "follow an EXERCISE box with its exercise code cell")
        for name in SOLUTION_LINE.findall(text):
            referenced.add(f"{folder}/solutions/{name}")
            if not SOLUTION_NAME.fullmatch(name):
                error(i, f"name solutions <topic>-<n>_<what>.py, not {name}")
            if not Path(folder, "solutions", name).exists():
                error(i, f"solutions/{name} does not exist")

tracked = subprocess.run(["git", "ls-files", "*/solutions/*.py"], capture_output=True, text=True, check=True).stdout.split()
for solution in tracked:
    if not solution.startswith("old_materials/") and solution not in referenced:
        errors.append(f"{solution}: no exercise names this solution")

print("\n".join(errors))
sys.exit(1 if errors else 0)
