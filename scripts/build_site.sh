#!/usr/bin/env bash
# Build the static dashboard into site/: the web app plus exported run data.
#
# Data comes from sample/runs.db when it is committed (real LLM runs cost
# money, so they are recorded once and checked in); otherwise a baseline run
# is generated here, which is free and deterministic.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf site
if [[ -f sample/runs.db ]]; then
  db=sample/runs.db
else
  db=$(mktemp -d)/runs.db
  uv run harness --db "$db" run --agents random,scripted \
    --tasks empty-6x6,lavagap-6x6,doorkey-6x6 --seeds 20 --run-id baselines >/dev/null
fi
uv run harness --db "$db" export --out site

npm --prefix web ci --no-audit --no-fund
npm --prefix web run build
cp -R web/dist/. site/
echo "site/ ready: $(find site -type f | wc -l | tr -d ' ') files, $(du -sh site | cut -f1)"
