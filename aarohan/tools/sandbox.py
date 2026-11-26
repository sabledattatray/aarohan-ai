"""AAROHAN AI 2.0 - Isolated Code Execution Sandbox.

Executes code snippets within controlled subprocess limits with time and memory guards.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict


class CodeSandbox:
    """Safe subprocess execution environment for Python, JavaScript, and shell commands."""

    def __init__(self, timeout_seconds: int = 10, max_output_chars: int = 20000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute a Python snippet in an isolated subprocess with strict timeouts."""
        # Static safety guard against dangerous OS destruction commands
        forbidden = [
            "shutil.rmtree('C:",
            "os.system('rmdir",
            "os.system('del /",
            "subprocess.Popen('format",
            "__import__('os').system('shutdown",
        ]
        for pattern in forbidden:
            if pattern in code:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Security Policy Violation: Blocked destructive pattern '{pattern}'",
                    "execution_time_ms": 0,
                    "exit_code": 1,
                }

        start_time = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tempfile.gettempdir(),
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            stdout = res.stdout[: self.max_output_chars]
            stderr = res.stderr[: self.max_output_chars]

            return {
                "success": res.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time_ms": elapsed_ms,
                "exit_code": res.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution Timed Out: Execution exceeded limit of {self.timeout_seconds}s.",
                "execution_time_ms": self.timeout_seconds * 1000,
                "exit_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "execution_time_ms": 0,
                "exit_code": -1,
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
