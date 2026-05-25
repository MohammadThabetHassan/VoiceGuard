#!/usr/bin/env bash
# Team commit helper — enforces conventional commits and correct author identity.
# Usage: team_commit.sh --m|-f|-a -m "message" [--co m,f|a]
#   --m | --f | --a  : author identity (Mohammad / Fahad / Ahmed)
#   -m "message"     : conventional commit message (required)
#   --co m,f         : add Co-authored-by trailers for cross-owner work

set -euo pipefail

CONVENTIONAL_RE='^(feat|fix|docs|chore|test|refactor|ci)(\(.+\))?: .+'

IDENTITY=""
MESSAGE=""
COAUTHORS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --m) IDENTITY="m"; shift ;;
    --f) IDENTITY="f"; shift ;;
    --a) IDENTITY="a"; shift ;;
    -m)  MESSAGE="$2"; shift 2 ;;
    --co)
      IFS=',' read -ra CO <<< "$2"
      for c in "${CO[@]}"; do
        case "$c" in
          m) COAUTHORS+=("Co-authored-by: Mohammad Thabet <MohammadThabetHassan@users.noreply.github.com>") ;;
          f) COAUTHORS+=("Co-authored-by: Fahad Sadek <fahadALjazzeri@users.noreply.github.com>") ;;
          a) COAUTHORS+=("Co-authored-by: Ahmed Alameri <AhmedSamiAlameri@users.noreply.github.com>") ;;
          *) echo "Unknown co-author: $c (use m, f, or a)"; exit 1 ;;
        esac
      done
      shift 2
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -z "$IDENTITY" ]] && { echo "Error: must specify --m, --f, or --a"; exit 1; }
[[ -z "$MESSAGE" ]]  && { echo "Error: must specify -m 'message'"; exit 1; }

if ! echo "$MESSAGE" | grep -qE "$CONVENTIONAL_RE"; then
  echo "Error: message does not match conventional commit format:"
  echo "  Expected: feat|fix|docs|chore|test|refactor|ci(scope): description"
  echo "  Got: $MESSAGE"
  exit 1
fi

case "$IDENTITY" in
  m) NAME="Mohammad Thabet"; EMAIL="MohammadThabetHassan@users.noreply.github.com" ;;
  f) NAME="Fahad Sadek";    EMAIL="fahadALjazzeri@users.noreply.github.com" ;;
  a) NAME="Ahmed Alameri";  EMAIL="AhmedSamiAlameri@users.noreply.github.com" ;;
esac

FULL_MESSAGE="$MESSAGE"
if [[ ${#COAUTHORS[@]} -gt 0 ]]; then
  FULL_MESSAGE="$MESSAGE"$'\n\n'
  for co in "${COAUTHORS[@]}"; do
    FULL_MESSAGE+="$co"$'\n'
  done
fi

git -c user.name="$NAME" -c user.email="$EMAIL" commit -m "$FULL_MESSAGE"
SHA=$(git rev-parse --short HEAD)
echo "Committed as $NAME <$EMAIL>"
echo "SHA: $SHA"

bash "$(dirname "$0")/append_progress.sh" "$SHA" "$MESSAGE"
