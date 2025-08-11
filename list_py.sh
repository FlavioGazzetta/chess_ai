#!/usr/bin/env bash
# Write name + full path and the content of each Python file in the current dir
# into project_python_content.txt (non-recursive).

set -euo pipefail

OUTFILE="project_python_content.txt"
: > "$OUTFILE"  # truncate/create

# Collect .py files safely (handles spaces/newlines)
mapfile -d '' -t files < <(find . -maxdepth 1 -type f -name "*.py" -print0)

if [ "${#files[@]}" -eq 0 ]; then
  printf "No Python files found in: %s\n" "$(pwd)" >> "$OUTFILE"
  exit 0
fi

for f in "${files[@]}"; do
  # Resolve a full path with fallbacks
  if command -v realpath >/dev/null 2>&1; then
    fullpath=$(realpath -- "$f")
  elif command -v readlink >/dev/null 2>&1; then
    fullpath=$(readlink -f -- "$f")
  else
    fullpath="$(pwd)/${f#./}"
  fi

  {
    echo "=============================="
    echo "File: $(basename "$f")"
    echo "Path: $fullpath"
    echo "------------------------------"
    cat -- "$f"
    echo
  } >> "$OUTFILE"
done

echo "Wrote $((${#files[@]})) file(s) to $OUTFILE"
