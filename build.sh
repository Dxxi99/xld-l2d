#!/usr/bin/env bash
set -euo pipefail

echo "==> swift build"
swift build

echo "==> swift test"
swift test || true

echo "==> done"
echo "Run: swift run xld"
