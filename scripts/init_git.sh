#!/usr/bin/env bash
# Create a local git repo inside MLOPS so ClearML Agent clones this project,
# not the parent ~/VSCode repository.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -d .git ]]; then
  echo "Git repo already exists in ${ROOT}"
else
  git init
  git config user.email "${GIT_AUTHOR_EMAIL:-mlops@local.dev}"
  git config user.name "${GIT_AUTHOR_NAME:-MLOPS}"
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit — repo is ready."
else
  git commit -m "MLOPS project snapshot for ClearML Agent"
  echo "Created commit: $(git rev-parse --short HEAD)"
fi

echo "Agent will clone: file://${ROOT}"
