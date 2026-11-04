"""Download the demo T1-weighted scans listed in demo_data.json.

The scans are from OpenNeuro ds000001, released under CC0 (no restrictions).
Each file is checked against the SHA-256 in the manifest, and files that are
already there and correct are not downloaded again.

    python fetch_demo_data.py                # all scans into /data
    python fetch_demo_data.py --out ./data --subjects 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from loguru import logger

import config

MANIFEST = Path(__file__).with_name("demo_data.json")
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected_sha256: str) -> bool:
    """Download `url` to `target` unless it is already there; True if a download happened."""
    if target.exists() and sha256(target) == expected_sha256:
        logger.info(f"{target.name}: already downloaded")
        return False
    partial = target.with_name(target.name + ".part")
    logger.info(f"{target.name}: downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as handle:
        while block := response.read(CHUNK):
            handle.write(block)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise RuntimeError(f"{target.name}: checksum mismatch (got {actual}); the source may have changed")
    partial.replace(target)
    return True


def main(argv: list[str] | None = None) -> int:
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=f"Download the demo scans ({manifest['license']}).")
    parser.add_argument("--out", type=Path, default=config.DEFAULT_DATA_DIR, help="target folder (default: %(default)s)")
    parser.add_argument("--subjects", type=int, default=len(manifest["files"]),
                        help="how many scans, from the start of the list (default: all %(default)s)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"][: args.subjects]:
        fetch(f"{manifest['base_url']}/{entry['path']}", args.out / Path(entry["path"]).name, entry["sha256"])
    logger.info(f"Data: {manifest['dataset']} ({manifest['license']}), {manifest['homepage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
