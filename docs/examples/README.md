# Example scores and templates

One real score, as the model wrote it for a 4-minute song, in each of the forms
[SCORE_RECIPES.md](../SCORE_RECIPES.md) describes. Every file here is a value for
a request's `"abc"` (plain scores) or `"abc_template"` (`*.tpl.abc`), with the
same lyrics the original was rendered with.

| file | request field | how it was made from `song.abc` |
|------|---------------|---------------------------------|
| `song.abc` | `abc` | the original, from a render's `--artifacts` dir |
| `song_tr+5_vo-1.abc` | `abc` | `lua scripts/abc_transpose.lua song.abc 5 -1`: Em → Am, singer an octave below that |
| `song_melody.abc` | `abc` + `"cot": "melody"` | chord symbols stripped: `sed -E '/^[A-Za-z]:/! s/"[^"]*"//g'` |
| `song_no_ins.abc` | `abc` | every Ins body line replaced by whole-bar rests of the same length (`Z4\|`); the accompaniment is improvised |
| `song_new_ins.tpl.abc` | `abc_template` | every Ins body line replaced by `%%yue2-gen`; the model writes an accompaniment under the kept vocal |
| `song_new_vocal.tpl.abc` | `abc_template` | every Vocal body line replaced by `%%yue2-gen bars=N` (N counted from the line it replaces); the model writes melody and chords over the kept accompaniment |
| `song_rewrite_keep_intro.tpl.abc` | `abc_template` | both voices are holes except in `% intro`; the model rewrites the song from the kept opening |
| `song_new_ins_primer.tpl.abc` | `abc_template` | `song_new_ins.tpl.abc` with the first verse's Vocal lines in a primer block before the intro (`yue2` warns about this one) |
| `song_continue.tpl.abc` | `abc_template` | the header, the intro's Ins lines with no Vocal lines under them, then `%%yue2-continue`; the model writes the rest of the score freely (and, given no chord symbols, without any) |
| `song_intro_chords.tpl.abc` | `abc_template` | the same intro Ins lines, each under a `V: Vocal` / `%%yue2-chords` pair, then `%%yue2-continue`; the model writes the intro's chord lines after seeing each Ins line, then the rest with chords |

The awk that made the `no_ins` and `new_vocal` files, with the bar count a
`ZN|` multi-rest stands for:

```awk
function bars(line,  n, i, m, seg) {
	n = 0; m = split(line, seg, "|")
	for (i = 1; i <= m; i++) if (seg[i] != "")
		n += (match(seg[i], /^Z[0-9]*$/) ? (length(seg[i]) > 1 ? substr(seg[i], 2) + 0 : 1) : 1)
	return n
}
/^V:/ { voice = $2 }
/^[A-Za-z]:/ || /^%/ || NF == 0 || voice != "Vocal" { print; next }
{ print "%%yue2-gen bars=" bars($0) }          # new_vocal; no_ins prints "Z" bars($0) "|" for voice == "Ins"
```

Render any of them with the lyrics in a request:

```bash
jq --rawfile abc docs/examples/song_new_ins.tpl.abc '. + {abc_template: $abc}' song.json > tpl.json
build/yue2 song --request tpl.json --out song.flac --artifacts out/tpl --gpu 0
```
