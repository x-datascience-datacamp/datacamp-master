"""Clear the outputs and execution counts of notebooks, and change nothing else.

Usage: python tools/strip_outputs.py [notebook.ipynb ...]
With no argument, strips every notebook tracked by git.
"""
import json
import subprocess
import sys
from pathlib import Path

paths = sys.argv[1:] or subprocess.run(
    ["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=True
).stdout.split()
for path in paths:
    text = Path(path).read_text()
    nb = json.loads(text)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    stripped = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if stripped != text:
        Path(path).write_text(stripped)
        print(f"stripped {path}")
