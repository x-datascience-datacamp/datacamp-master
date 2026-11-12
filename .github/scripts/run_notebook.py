"""Fill a notebook's exercise cells from its solutions/ and execute it.

Usage: python run_notebook.py <notebook.ipynb> <output_dir>
"""
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

SOLUTION = re.compile(r"solutions/[\w./-]+\.py")
LOAD = re.compile(r"#?\s*%load\s+(\S+)\s*")

path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
nb = nbformat.read(path, as_version=4)

# A solution referenced in a markdown cell fills the next empty or `...`
# skeleton code cell; a cell holding only `%load <file>` gets that file.
pending = []
for cell in nb.cells:
    src = cell.source.strip()
    if cell.cell_type == "markdown":
        pending += SOLUTION.findall(cell.source)
        continue
    load = LOAD.fullmatch(src)
    if load:
        pending = [load.group(1)]
    elif src and all(line.startswith("#") for line in src.splitlines()):
        continue
    elif src and "..." not in src:
        pending = []
    if pending:
        cell.source = "\n\n".join(
            f"# --- solution: {s}\n" + (path.parent / s).read_text()
            for s in pending
        )
        pending = []

client = NotebookClient(
    nb, timeout=None, kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
)
start = time.time()
try:
    client.execute()
    status, line = 0, f"✅ `{path}`"
except CellExecutionError as e:
    print(e)
    status, line = 1, f"❌ `{path}`: {e.ename}: {e.evalue}"

out_dir.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out_dir / path.name)
with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as f:
    f.write(f"{line} ({time.time() - start:.0f} s)\n")
sys.exit(status)
