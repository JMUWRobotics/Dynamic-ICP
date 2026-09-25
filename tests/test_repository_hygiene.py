import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_DIRECTORIES = {
    ".agents",
    ".codex",
    ".git",
    ".pytest_cache",
    ".venv",
    "Open3D",
    "__pycache__",
    "build",
    "dist",
    "dynamic_icp.egg-info",
    "outputs",
}


def maintained_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {
            ".dockerignore",
            ".gitattributes",
            ".gitignore",
            ".gitmodules",
            "CMakeLists.txt",
            "Dockerfile",
            "LICENSE",
        }:
            yield path


def test_public_files_do_not_contain_private_machine_paths():
    private_paths = re.compile(
        r"/" + r"(?:home|Users|media)/[^/\s]+/"
        r"|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"
    )
    findings = []
    for path in maintained_text_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if private_paths.search(line):
                findings.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert not findings, "private machine paths found in: " + ", ".join(findings)


def test_public_source_and_scripts_do_not_contain_chinese_comments_or_strings():
    han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    findings = []
    for path in maintained_text_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if han.search(line):
                findings.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert not findings, "Chinese text found in public files: " + ", ".join(findings)
