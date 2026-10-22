# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch the published checkpoint and the smallest official nuScenes release."""

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
DATA_SHA256 = "943037abbb3b26b3070dc76504a43eb440503b00baf9ac2f1538d9c03fc9298f"
CHECKPOINT_URL = "https://drive.google.com/uc?id=1q49_7aR7EWWOCAUK8iE5jfj1JdtL9gJy"
CHECKPOINT_SHA256 = "4488f70ce58d2313b21a5cf26fa4dee96599d4683a4bb225814da6f4d90ef4f2"
CHUNK = 64 * 1024 * 1024


def download_mini():
    destination = ROOT / "data/v1.0-mini.tgz"
    size = int(requests.head(DATA_URL, timeout=30).headers["Content-Length"])
    if destination.exists() and destination.stat().st_size == size:
        return destination
    parts = ROOT / "data/.mini_parts"
    parts.mkdir(parents=True, exist_ok=True)
    prefix = destination.with_suffix(".tgz.partial")
    if prefix.exists():
        with prefix.open("rb") as source:
            for index in range(prefix.stat().st_size // CHUNK):
                path = parts / f"{index:04d}"
                if not path.exists():
                    path.write_bytes(source.read(CHUNK))
                else:
                    source.seek(CHUNK, 1)

    def fetch(index):
        start, end = index * CHUNK, min((index + 1) * CHUNK, size) - 1
        path = parts / f"{index:04d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                with requests.get(
                    DATA_URL,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 90),
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                    ):
                        raise ValueError("Server did not honor the requested byte range")
                    with path.with_suffix(".partial").open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if path.with_suffix(".partial").stat().st_size != end - start + 1:
                    raise ValueError("Incomplete dataset chunk")
                path.with_suffix(".partial").replace(path)
                return
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    indices = range((size + CHUNK - 1) // CHUNK)
    print(
        f"Downloading nuScenes mini: {size / 1e9:.2f} GB, resumable in {len(indices)} chunks",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        for completed, future in enumerate(
            as_completed([pool.submit(fetch, index) for index in indices]), 1
        ):
            future.result()
            print(f"Dataset chunks: {completed}/{len(indices)}", flush=True)
    with destination.with_suffix(".tgz.partial").open("wb") as output:
        for index in indices:
            with (parts / f"{index:04d}").open("rb") as stream:
                shutil.copyfileobj(stream, output)
    destination.with_suffix(".tgz.partial").replace(destination)
    shutil.rmtree(parts)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["data", "checkpoint"])
    args = parser.parse_args()
    if args.asset == "checkpoint":
        import gdown

        path = ROOT / "weights/torch/epoch_20_ema.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            gdown.download(CHECKPOINT_URL, str(temporary), quiet=True)
            temporary.replace(path)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Published checkpoint checksum mismatch")
        print(f"Checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    folder = ROOT / "data/nuscenes"
    if (folder / "download.json").exists():
        print(f"nuScenes RGB keyframes already present: {folder}")
        return
    archive = download_mini()
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("Official nuScenes mini checksum mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            path = Path(member.name)
            wanted = (path.parts[0] == "v1.0-mini" and path.suffix == ".json") or (
                path.parts[0] == "samples"
                and len(path.parts) == 3
                and path.parts[1].startswith("CAM_")
                and path.suffix == ".jpg"
            )
            if member.isfile() and wanted:
                target = folder / path
                if not target.resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Unexpected dataset member path")
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
    if count != 2437:
        raise ValueError(f"Expected 2424 camera images and 13 metadata files, got {count}")
    (folder / "download.json").write_text(
        json.dumps(
            {
                "url": DATA_URL,
                "archive_sha256": digest,
                "files": count,
                "retained": "mini metadata and six-camera RGB keyframes only; no LiDAR, radar, maps or sweeps",
            },
            indent=2,
        )
        + "\n"
    )
    archive.unlink()
    print(f"Saved {count} camera/metadata files to {folder}", flush=True)


if __name__ == "__main__":
    main()
