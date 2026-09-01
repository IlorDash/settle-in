#!/usr/bin/env python3
"""Save the commit message on a failed check, restore it on the next commit.

- Called with no args (by the gitlint runner on failure): back up
  .git/COMMIT_EDITMSG so the message is not lost.
- Called with --restore (prepare-commit-msg stage): if a backup exists, put it
  back so the user does not have to retype their message.
"""

import shutil
import sys
from pathlib import Path

MSG = Path(".git/COMMIT_EDITMSG")
BAK = Path(".git/COMMIT_EDITMSG.bak")


def main():
    if "--restore" in sys.argv:
        if BAK.exists():
            shutil.copy(BAK, MSG)
            BAK.unlink()
            print("Restored your previous commit message.")
    elif MSG.exists():
        shutil.copy(MSG, BAK)


if __name__ == "__main__":
    main()
