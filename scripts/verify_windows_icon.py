from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pefile


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that a Windows EXE contains an icon resource.")
    parser.add_argument("executable", type=Path)
    executable = parser.parse_args().executable
    if not executable.is_file() or executable.stat().st_size == 0:
        parser.error(f"Windows executable not found: {executable}")

    with pefile.PE(str(executable), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            print(f"No PE resource directory in {executable}", file=sys.stderr)
            return 1
        resource_types = {entry.id for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries}

    missing = {3, 14} - resource_types  # RT_ICON and RT_GROUP_ICON
    if missing:
        print(f"Missing Windows icon resource types {sorted(missing)} in {executable}", file=sys.stderr)
        return 1
    print(f"Embedded Windows icon verified: {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
