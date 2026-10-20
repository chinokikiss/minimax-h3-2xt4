"""Check the frozen user environment; never provision or switch environments."""
from pathlib import Path
import sys
from patches.legacy_runtime import validate_packages,validate_source


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    validate_source(root/'ComfyUI')
    validate_packages()
    print('FROZEN_RUNTIME_OK',sys.executable)
    print('No packages were installed, upgraded, or downgraded.')
