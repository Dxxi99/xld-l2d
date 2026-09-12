#!/usr/bin/env python3
"""Depth Anything V2 CoreML (Apple) 다운로드."""
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("pip install huggingface_hub"); sys.exit(1)

REPO = "apple/coreml-depth-anything-v2-small"
TARGET = os.path.expanduser("~/XLD-L2D/models/depth-anything-v2-small")

print(f"downloading {REPO}")
print(f"target: {TARGET}")
os.makedirs(TARGET, exist_ok=True)

path = snapshot_download(
    repo_id=REPO,
    local_dir=TARGET,
    allow_patterns=["*.mlpackage/*", "*.mlpackage", "*.json", "*.md"],
)
print(f"downloaded to {path}")
for root, dirs, files in os.walk(TARGET):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), TARGET)
        size = os.path.getsize(os.path.join(root, f))
        print(f"  {rel}  ({size:,} bytes)")
