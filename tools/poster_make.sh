#!/bin/sh
# Rebuild the poster from the committed tables: the figures (SVG), their poster-type PNGs, then the PowerPoint file.
#   NODE_PATH=<a node_modules with pptxgenjs> PPTX_APPLY_THEME=<path to the pptx skill's apply_theme.js> tools/poster_make.sh
# Needs rsvg-convert (brew install librsvg) and node.
set -e
cd "$(dirname "$0")/.."
uv run tools/poster_figures.py                                             # docs/poster/*.svg, for reading on a screen
TMP=$(mktemp -d)
uv run tools/poster_figures.py --scale 1.3 --surface "#F4F4F1" --out "$TMP"  # poster type, on the card colour
mkdir -p docs/poster/png
for f in "$TMP"/*.svg; do
  n=$(basename "${f%.svg}")
  case "$n" in f4*) w=2400 ;; *) w=4200 ;; esac                            # about 300 px per inch when placed
  rsvg-convert -w "$w" "$f" -o "docs/poster/png/$n.png"
done
rm -r "$TMP"
node tools/poster_build.js
