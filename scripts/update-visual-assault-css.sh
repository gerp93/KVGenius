#!/usr/bin/env bash
# Re-vendors src/renderer/themes.css from gerp93/VisualAssault's packages/css/themes.css.
# Usage: scripts/update-visual-assault-css.sh <tag>   (e.g. v0.2.1)
set -euo pipefail

TAG="${1:?Usage: $0 <tag> (e.g. v0.2.1)}"
OUT="$(dirname "$0")/../src/renderer/themes.css"
URL="https://raw.githubusercontent.com/gerp93/VisualAssault/${TAG}/packages/css/themes.css"

{
  echo "/* Vendored from gerp93/VisualAssault packages/css/themes.css @ ${TAG}"
  echo " * Re-vendor with: scripts/update-visual-assault-css.sh <tag> */"
  echo ""
  curl -fsSL "$URL"
} > "$OUT"

echo "Updated $OUT from VisualAssault @ ${TAG}"
