#!/usr/bin/env bash
#set -euo pipefail

# link_if_missing <TARGET> <LINK>
link_if_missing() {
  local target="$1" link="$2"
  mkdir -p "$(dirname "$link")"

  if [ -L "$link" ]; then
    # Link exists: check its target
    local current
    current="$(readlink "$link")" || current=""
    if [ "$current" = "$target" ]; then
      echo "ok: $link -> $current"
      return 0
    else
      echo "update: $link (was $current) -> $target"
      ln -sfn "$target" "$link"
      return 0
    fi
  elif [ -e "$link" ]; then
    echo "error: $link exists and is not a symlink" >&2
    return 1
  else
    ln -s "$target" "$link"
    echo "created: $link -> $target"
  fi
}

