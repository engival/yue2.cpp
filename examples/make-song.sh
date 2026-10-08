#!/bin/bash
# Make one song. Edit the settings between the two rules, then run:  ./make-song.sh
# (Windows: make-song.ps1 + make-song.bat are the same thing.)
# The song lands in a directory named after NAME, next to this script:
#   NAME/NAME.flac      the song
#   NAME/lyrics.txt     the lyrics as sent
#   NAME/request.json   the request as sent
#   NAME/render/        what the model wrote (score.abc, ...) - keep it to reuse the tune
# ---------------------------------------------------------------------------

NAME="first-song"
SEED=1		# another number = another song from the same words
GPU=0		# which graphics card; 0 = the first one

# Describe the sound: genre, instruments, singer. Broad common words work best.
STYLE="slow dream pop, reverb guitar, brushed drums, soft female vocal"

# [Verse], [Chorus], [Bridge], [Outro] ... on their own lines, a blank line between parts.
LYRICS=$(cat <<'EOF'
[Verse]
The kettle sings a flat blue note
The window holds the evening in
Your coat is drying by the door
And the radio is whispering

[Chorus]
Stay a while, the rain is warm
Stay a while, we'll ride the storm
Nothing out there needs us now
Stay a while

[Verse]
The streetlights blur on the glass
A dog is barking down the lane
We count the seconds after thunder
And we laugh and count again

[Chorus]
Stay a while, the rain is warm
Stay a while, we'll ride the storm
Nothing out there needs us now
Stay a while
EOF
)

# Extras - leave empty ("") to skip, fill in to try one.
TEMPO=""		# beats per minute, 20-300, e.g. 90
AVOID=""		# a sound to steer away from, e.g. "minimal, repetitive, drone, ambient"
SWITCH_STYLE=""		# change to this style part-way, e.g. "90s eurodance, four-on-the-floor beat, supersaw synths, powerful female vocal"
SWITCH_AT_CHORUS=2	# ... from this chorus on (the song is rendered twice)
SAME_TUNE_AS=""		# NAME of an earlier song: sing its tune in this STYLE (keep the same LYRICS, pick a new NAME)

# ---------------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1

YUE2=""
for f in ../yue2 ../build/yue2 ./yue2
do
	if [ -x "$f" ]; then YUE2=$f; break; fi
done
if [ -z "$YUE2" ]; then echo "cannot find the yue2 binary in ../ or ../build/" >&2; exit 1; fi

# A JSON string: quotes, backslashes and control characters escaped.
json_str()
{
	local s=${1//\\/\\\\}
	s=${s//\"/\\\"}
	s=${s//$'\t'/\\t}
	s=${s//$'\n'/\\n}
	s=${s//$'\r'/}
	printf '"%s"' "$s"
}

mkdir -p "$NAME" || exit 1
printf '%s\n' "$LYRICS" > "$NAME/lyrics.txt"
{
	echo "{"
	echo "	\"style\": $(json_str "$STYLE"),"
	echo "	\"lyrics_file\": \"lyrics.txt\","
	if [ -n "$TEMPO" ]; then echo "	\"score_tempo\": $TEMPO,"; fi
	if [ -n "$AVOID" ]
	then
		echo "	\"cfg_scale\": 3,"
		echo "	\"negative_style\": $(json_str "$AVOID"),"
		echo "	\"negative_lyrics\": true,"
	fi
	if [ -n "$SWITCH_STYLE" ]
	then
		echo "	\"handover\": [ { \"section\": \"chorus\", \"nth\": $SWITCH_AT_CHORUS, \"style\": $(json_str "$SWITCH_STYLE") } ],"
	fi
	if [ -n "$SAME_TUNE_AS" ]; then echo "	\"abc_file\": $(json_str "../$SAME_TUNE_AS/render/score.abc"),"; fi
	echo "	\"seed\": $SEED"
	echo "}"
} > "$NAME/request.json"

"$YUE2" song --request "$NAME/request.json" --out "$NAME/$NAME.flac" --artifacts "$NAME/render" --gpu "$GPU" \
	&& echo "done: $(pwd)/$NAME/$NAME.flac"
