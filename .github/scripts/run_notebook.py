"""Execute a notebook with its exercise cells loaded from solutions/.

An exercise cell starts with `# %load solutions/<file>.py`: like running
`%load`, the cell is replaced by that file before the notebook runs.

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

LOAD = re.compile(r"#\s*%load\s+(\S+)")

path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
nb = nbformat.read(path, as_version=4)
for cell in nb.cells:
    load = LOAD.fullmatch(cell.source.split("\n", 1)[0].strip())
    if cell.cell_type == "code" and load:
        cell.source = (path.parent / load.group(1)).read_text()

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
