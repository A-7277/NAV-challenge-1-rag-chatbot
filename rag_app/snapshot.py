"""Export or restore a portable Qdrant collection snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from rag_app.config import Settings


def export_snapshot(settings: Settings, output_dir: Path) -> Path:
    base = settings.qdrant_url.rstrip("/")
    response = requests.post(
        f"{base}/collections/{settings.qdrant_collection}/snapshots", timeout=180
    )
    response.raise_for_status()
    name = response.json()["result"]["name"]
    download = requests.get(
        f"{base}/collections/{settings.qdrant_collection}/snapshots/{name}",
        timeout=300,
    )
    download.raise_for_status()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    destination.write_bytes(download.content)
    metadata = {
        "collection": settings.qdrant_collection,
        "snapshot": name,
        "embedding_model": settings.embedding_model,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return destination


def restore_snapshot(settings: Settings, snapshot: Path) -> None:
    base = settings.qdrant_url.rstrip("/")
    with snapshot.open("rb") as handle:
        response = requests.post(
            f"{base}/collections/{settings.qdrant_collection}/snapshots/upload",
            params={"priority": "snapshot"},
            files={"snapshot": (snapshot.name, handle, "application/octet-stream")},
            timeout=600,
        )
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--output", type=Path, default=Path("artifacts/snapshots"))
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    settings = Settings.load()
    if args.command == "export":
        print(export_snapshot(settings, args.output))
    else:
        restore_snapshot(settings, args.snapshot)
        print(f"Restored {args.snapshot}")


if __name__ == "__main__":
    main()

