#!/bin/bash
# Ship the working tree to the pod through the PTY-only proxy (wrapped base64; a PTY line holds ~4 KB)
# and (re)install mtedit into the pod venv. Usage: scripts/sync.sh
set -e
cd "$(dirname "$0")/.."
B64=$(git ls-files -co --exclude-standard | grep -v '^chess_test/image' | tar czf - -T - | base64 -b 76)
{ echo 'rm -rf /workspace/multi_turn && mkdir -p /workspace/multi_turn && cd /workspace/multi_turn'
  echo "base64 -d > /tmp/mt.tgz <<'B64END'"; echo "$B64"; echo 'B64END'
  echo 'tar xzf /tmp/mt.tgz && uv pip install -q --python /workspace/qi/bin/python -e . --no-deps && echo SYNC_OK $(git ls-files 2>/dev/null | wc -l) $(ls | wc -l)'
} | scripts/rp | grep -E "SYNC_OK|rror" || true
