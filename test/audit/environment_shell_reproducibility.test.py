from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OK_MARKER = "[OK] manager-import"


def _write_env_check_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "shell_env_check.py"
    script_path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from relic.parameters.manager import get_global_manager\n"
        "get_global_manager()\n"
        "print('[OK] manager-import')\n",
        encoding="utf-8",
    )
    return script_path


def _run_shell(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _quote_powershell(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def test_environment_check_is_reproducible_in_cmd_and_powershell(tmp_path: Path) -> None:
    script_path = _write_env_check_script(tmp_path)
    python_path = str(Path(sys.executable).resolve())
    script_arg = str(script_path.resolve())

    cmd_line = subprocess.list2cmdline([python_path, script_arg])
    cmd_result = _run_shell(["cmd", "/d", "/s", "/c", cmd_line])
    cmd_out = f"{cmd_result.stdout}\n{cmd_result.stderr}"
    assert cmd_result.returncode == 0, f"cmd failed: {cmd_out}"
    assert OK_MARKER in cmd_out, f"cmd missing marker: {cmd_out}"

    ps_line = (
        f"& {_quote_powershell(python_path)} "
        f"{_quote_powershell(script_arg)}"
    )
    ps_result = _run_shell(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_line]
    )
    ps_out = f"{ps_result.stdout}\n{ps_result.stderr}"
    assert ps_result.returncode == 0, f"powershell failed: {ps_out}"
    assert OK_MARKER in ps_out, f"powershell missing marker: {ps_out}"
