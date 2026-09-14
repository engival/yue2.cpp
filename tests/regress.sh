#!/bin/bash
# tests/regress.sh — decode every song in the reference-song render set through
# the standalone, exact-F32 `yue2-vae` and score each against its audio.flac.
# This is the exact-decode reference path: `yue2 song` renders at the NAR's
# fp16-staged precision instead (SPEC_SINGLE.md §2.2), so it is not what this
# scores.
#
# Usage:
#	tests/regress.sh                # full run, all songs
#	tests/regress.sh alley_swing_s1 # single song
#
# YUE2_REGRESS_DEVICE (cpu|vulkan, default vulkan) and YUE2_REGRESS_GPU
# (default 0) pick the backend — device 0 is the AMD 7900 XTX, 1 the Arc B70.
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
DEVICE=${YUE2_REGRESS_DEVICE:-vulkan}
GPU=${YUE2_REGRESS_GPU:-0}

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

# An empty set used to exit 0, which reads as "everything passed".
if [ "${#names[@]}" -eq 0 ]; then
	echo "no <name>/{latent.npy,audio.flac} pairs under $SONGS_DIR — set YUE2_REGRESS_SONGS" >&2
	exit 1
fi

fail=0
for name in "${names[@]}"; do
	latent="$SONGS_DIR/$name/latent.npy"
	flac="$SONGS_DIR/$name/audio.flac"
	npy_out="$OUT_DIR/$name.npy"

	echo "=== $name ===" | tee -a "$LOG"
	"$BIN" -m "$GGUF" -i "$latent" --npy "$npy_out" \
		--device "$DEVICE" --gpu "$GPU" --core-frames 256 2>&1 | tee -a "$LOG"
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
