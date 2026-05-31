#!/usr/bin/env bash
# Download Norwegian law datasets from Lovdata's free public API.
# New packages are published nightly — always the latest version of Norwegian law.
# No authentication required.

set -euo pipefail

DATA_DIR="${1:-./data/norwegian-law}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading gjeldende-lover (current laws)..."
curl -fSL -O https://api.lovdata.no/v1/publicData/get/gjeldende-lover.tar.bz2

echo "Downloading gjeldende-sentrale-forskrifter (current central regulations)..."
curl -fSL -O https://api.lovdata.no/v1/publicData/get/gjeldende-sentrale-forskrifter.tar.bz2

echo "Extracting..."
tar xjf gjeldende-lover.tar.bz2
tar xjf gjeldende-sentrale-forskrifter.tar.bz2

echo "Done. Data extracted to: $DATA_DIR"
echo "  - gjeldende-lover/"
echo "  - gjeldende-sentrale-forskrifter/"
