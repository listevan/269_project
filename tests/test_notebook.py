import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "token_replay_pilot.ipynb"


def test_notebook_is_current_and_parses(tmp_path):
    before = NB.read_text()
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py")], check=True, capture_output=True)
    assert NB.read_text() == before, "notebook is stale: run tools/build_notebook.py and commit"
    nb = json.loads(before)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_flags_match_scripts():
    """Every --flag the notebook passes must exist in the script it calls."""
    import re
    src = "\n".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code") + "\n"
    calls = re.findall(r'run\("(\w+\.py)",(.*?)\)\n', src, flags=re.S)
    assert [c[0] for c in calls] == ["rollout.py", "score.py", "replay.py"]
    for script, args in calls:
        text = (ROOT / "src" / script).read_text()
        for flag in re.findall(r'"(--[\w-]+)"', args):
            assert f'"{flag}"' in text, f"{script} has no {flag}"
