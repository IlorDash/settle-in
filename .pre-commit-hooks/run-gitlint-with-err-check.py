#!/usr/bin/env python3
"""Run gitlint on the commit message; save the message if it fails."""

import os
import subprocess
import sys

CONFIG_FILE = ".gitlint"
COMMIT_MSG_FILE = ".git/COMMIT_EDITMSG"


def main():
    print(f"Running gitlint with config: {CONFIG_FILE}")

    result = subprocess.run(
        ["gitlint", f"--config={CONFIG_FILE}", f"--msg-filename={COMMIT_MSG_FILE}"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    if result.returncode != 0:
        # Use the current interpreter so this works on Windows (no `python3`).
        os.system(f'"{sys.executable}" .pre-commit-hooks/save-restore-commit-msg.py')
        print()
        print(
            "gitlint failed. Saved your commit message; it will be restored on the "
            "next 'git commit'."
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
