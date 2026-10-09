"""Build a checksummed source-and-artifact bundle for the Windows demo laptop."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "transfer",
}


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def build_bundle(project_root: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-transfer-") as temporary:
        bundle_root = Path(temporary) / "traceable-pdf-rag"
        for source in project_root.rglob("*"):
            relative = source.relative_to(project_root)
            if any(part in EXCLUDED_PARTS for part in relative.parts) or source.name == ".env":
                continue
            target = bundle_root / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        entries = []
        for file_path in sorted(path for path in bundle_root.rglob("*") if path.is_file()):
            if file_path.name == "SHA256SUMS":
                continue
            relative = file_path.relative_to(bundle_root).as_posix()
            entries.append(f"{_digest(file_path)}  {relative}")
        (bundle_root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
        archive_base = destination.with_suffix("")
        archive = shutil.make_archive(
            str(archive_base), "zip", bundle_root.parent, bundle_root.name
        )
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("transfer/traceable-pdf-rag.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(build_bundle(root, args.output))


if __name__ == "__main__":
    main()
