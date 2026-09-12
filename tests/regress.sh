#!/bin/bash
# tests/regress.sh — decode every song in the reference-song render set on Vulkan
# device 0 (AMD 7900 XTX) and score each against its audio.flac.
#
# Usage:
#	tests/regress.sh                # full run, all songs
#	tests/regress.sh alley_swing_s1 # single song
#
# Writes decoded .npy to tests/out/regress/<name>.npy and a combined log to
# tests/out/regress/regress.log. songs/ is never written to.
set -uo pipefail

cd "$(dirname "$0")/.."

# Both default to the sibling workspace next to this repo; override in the
# environment to point them elsewhere. The author's own set is 45 private
# renders of a single reference song, one directory per style; any directory
# of <name>/{latent.npy,audio.flac} pairs works.
SONGS_DIR=${YUE2_REGRESS_SONGS:-"$(dirname "$0")/../../songs/ref_song/out"}
GGUF=tests/out/yue2-vae-f32.gguf
BIN=build/yue2-vae
PY=${PY:-../venv_yue2/bin/python}
OUT_DIR=tests/out/regress

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/regress.log"
: > "$LOG"

if [ "$#" -ge 1 ]; then
	names=("$@")
else
	names=()
	for d in "$SONGS_DIR"/*/; do
		d="${d%/}"
		[ -f "$d/latent.npy" ] && [ -f "$d/audio.flac" ] && names+=("$(basename "$d")")
	done
fi

fail=0
for name in "${names[@]}"; do
	latent="$SONGS_DIR/$name/latent.npy"
	flac="$SONGS_DIR/$name/audio.flac"
	npy_out="$OUT_DIR/$name.npy"

	echo "=== $name ===" | tee -a "$LOG"
	"$BIN" -m "$GGUF" -i "$latent" --npy "$npy_out" \
		--device vulkan --gpu 0 --core-frames 256 2>&1 | tee -a "$LOG"
	decode_status=${PIPESTATUS[0]}

	if [ "$decode_status" -ne 0 ]; then
		echo "!!! $name: decode FAILED (exit $decode_status)" | tee -a "$LOG"
		fail=1
		continue
	fi

	"$PY" convert/compare.py "$flac" "$npy_out" 2>&1 | tee -a "$LOG"
	compare_status=${PIPESTATUS[0]}
	if [ "$compare_status" -ne 0 ]; then
		echo "!!! $name: compare.py exited $compare_status" | tee -a "$LOG"
	fi
done

exit $fail
