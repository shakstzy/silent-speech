#!/usr/bin/env bash
# Usage: find_videos.sh [per_query_count=15] ; writes data_prep/urls.txt (CC-BY licensed, English talking-to-camera).
# Uses YouTube's Creative Commons search filter (sp=EgIwAQ) AND a yt-dlp license match-filter.
set -uo pipefail
N=${1:-15}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/urls.txt"
QUERIES=("vlog talking to camera" "my day vlog update talking" "tutorial talking head webcam" "interview one on one webcam" "story time talking to camera" "q&a talking to the camera" "how to explain webcam" "my thoughts talking head" "podcast solo monologue webcam" "review talking to camera")
YT="python3 $HERE/yt.py"   # serial, cookies/anonymous/refresh order; exit 75 = bot check or 429, stop everything
$YT --guard || exit 75
yt() { "$@"; rc=$?; [ $rc -eq 75 ] && { echo "stopped: see $HERE/.blocked" >&2; exit 75; }; return 0; }
tmp=$(mktemp); : > "$tmp"
for q in "${QUERIES[@]}"; do
  enc=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote_plus(sys.argv[1]))' "$q")
  yt $YT --flat-playlist --playlist-end $((N*3)) --print "%(id)s" \
    "https://www.youtube.com/results?search_query=${enc}&sp=EgIwAQ%253D%253D" > "$tmp.q" 2>/dev/null
  head -n $((N*3)) "$tmp.q" | while read -r id; do echo "https://www.youtube.com/watch?v=$id"; done >> "$tmp"
done
sort -u "$tmp" > "$tmp.u"
# verify license, language, duration (2-25 min), one video at a time
: > "$tmp.o"
while read -r u; do
  yt $YT --skip-download --no-warnings --match-filter "license*=Creative & duration>=120 & duration<=1500 & language?=en" \
    --print "%(webpage_url)s" "$u" >> "$tmp.o" 2>/dev/null
done < "$tmp.u"
awk '!s[$0]++' "$tmp.o" > "$OUT"
rm -f "$tmp" "$tmp.u" "$tmp.q" "$tmp.o"
echo "wrote $(wc -l < "$OUT") urls to $OUT"
