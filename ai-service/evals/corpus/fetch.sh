#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
urls=(
  https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf
  https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf
  https://www.irs.gov/pub/irs-pdf/p15.pdf
  https://www.irs.gov/pub/irs-pdf/p463.pdf
  https://www.irs.gov/pub/irs-pdf/p587.pdf
)
for u in "${urls[@]}"; do
  f=$(basename "$u")
  [ -s "$f" ] || curl -fsSL -o "$f" "$u"
  echo "$f"
done
