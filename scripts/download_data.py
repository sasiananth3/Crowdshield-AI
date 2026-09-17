"""Download cited academic sources with checksums; no credentials or mirrors."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "shanghaitech": (
        "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1",
        "ShanghaiTech.zip",
        "https://github.com/desenzhou/ShanghaiTechDataset",
    ),
    "umn_demo": (
        "https://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi",
        "umn-demo.avi",
        "https://mha.cs.umn.edu/proj_events.shtml",
    ),
}


def download(name):
    url, filename, source = SOURCES[name]
    directory = ROOT / "data" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        part = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            url, headers={"User-Agent": "CrowdShield-Academic-Prototype/0.1"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            part.open("wb") as out,
        ):
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise RuntimeError(f"{name}: source returned an HTML page, not data")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 2_000_000_000:
                    raise RuntimeError("Download exceeds the 2 GB safety limit")
                out.write(chunk)
        if part.stat().st_size < 10_000:
            raise RuntimeError(f"{name}: suspiciously small download")
        part.replace(target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if filename.endswith(".zip"):
        destination = directory / "ShanghaiTech"
        with zipfile.ZipFile(target) as archive:
            members = [
                m
                for m in archive.infolist()
                if "part_B" in m.filename or "part_B_final" in m.filename
            ]
            if not members:
                raise RuntimeError("Archive does not contain ShanghaiTech Part B")
            if sum(m.file_size for m in members) > 2_000_000_000:
                raise RuntimeError("Uncompressed data exceeds 2 GB")
            for member in members:
                resolved = (destination / member.filename).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise RuntimeError("Unsafe archive path")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Symlinks are not accepted")
            archive.extractall(destination, members=members)
    receipt = {
        "dataset": name,
        "source": source,
        "download_url": url,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (directory / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(SOURCES))
    args = parser.parse_args()
    names = [args.only] if args.only else list(SOURCES)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, names))
