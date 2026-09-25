"""Shared process and path helpers for repository command-line scripts."""

from pathlib import Path
import sys


def module_command(module, arguments):
    return [sys.executable, "-m", module, *[str(value) for value in arguments]]


def require_directory(path, label):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{label} directory does not exist: {path}")
    return path


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    return path
