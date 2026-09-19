-- abc_transpose.lua — transpose a YuE2 score.abc. Plain Lua 5.3+, no modules.
--
--   lua scripts/abc_transpose.lua score.abc SEMITONES [VOCAL_OCTAVES] > new.abc
--   lua scripts/abc_transpose.lua score.abc 5 -1      Em -> Am, singer an octave down from that
--
-- Loaded with dofile()/require instead, it returns
--   transpose(abc, semitones, vocal_octaves) -> new abc, "Em -> Am" description
--
-- Every note of both voices moves `semitones`, the chord symbols and the K:
-- field move with them, and the `V: Vocal` notes move a further
-- `vocal_octaves` * 12. Notes are read to absolute pitch (key signature +
-- accidentals that last to the bar line) and respelled in the new key, so a
-- raised seventh stays a raised seventh. The result is re-read and compared
-- pitch by pitch before it is returned; a mismatch raises.
--
-- Scope = what the planner writes (surveyed over 340 scores): one K: in the
-- header, major or minor only, bodies of notes / rests / bar lines / ties, and
-- "chord symbols" of root + suffix [+ /bass]. Key names follow its habits
-- (D#m, G#m, Bbm; Db, Eb, F#, Ab, Bb).

local LETTER_PC = { C = 0, D = 2, E = 4, F = 5, G = 7, A = 9, B = 11 }
local SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
local MAJOR = { [0] = "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B" }
local MINOR = { [0] = "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B" }
-- sharps (+) or flats (-) in the signature of the major key on each pitch class
local MAJOR_SHARPS = { [0] = 0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5 }
local SHARP_NAMES = { [0] = "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }
local FLAT_NAMES  = { [0] = "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B" }

local function name_pc(name)
	return (LETTER_PC[name:sub(1, 1)] + (name:sub(2) == "#" and 1 or name:sub(2) == "b" and -1 or 0)) % 12
end

-- A key: its tonic pitch class, mode, and the signature as letter -> -1/0/+1.
local function make_key(pc, minor)
	local sharps = MAJOR_SHARPS[(pc + (minor and 3 or 0)) % 12]
	local key = { pc = pc, minor = minor, sharps = sharps, sig = {}, name = (minor and MINOR or MAJOR)[pc] .. (minor and "m" or "") }
	for letter in pairs(LETTER_PC) do key.sig[letter] = 0 end
	for i = 1, math.abs(sharps) do
		key.sig[(sharps > 0 and SHARP_ORDER or FLAT_ORDER):sub(i, i)] = sharps > 0 and 1 or -1
	end
	return key
end

local function parse_key(field)
	local tonic, mode = field:match("^K:%s*([A-G][#b]?)%s*(%a*)")
	assert(tonic and (mode == "" or mode == "m" or mode == "min" or mode == "maj"), "unsupported key field '" .. field .. "'")
	return make_key(name_pc(tonic), mode == "m" or mode == "min")
end

local ACC_VALUE = { ["^"] = 1, ["^^"] = 2, ["_"] = -1, ["__"] = -2, ["="] = 0 }
local ACC_TEXT  = { [1] = "^", [2] = "^^", [-1] = "_", [-2] = "__", [0] = "=" }

-- Spelling of MIDI pitch m in `key`: the scale's own letter when there is one,
-- else a natural, else the minor leading tone as a raised seventh, else a
-- sharp in sharp keys and a flat in flat keys.
local function spell(m, key)
	local pc = m % 12
	local tries = {}
	for letter, base in pairs(LETTER_PC) do
		if (base + key.sig[letter]) % 12 == pc then tries[#tries + 1] = { letter, key.sig[letter], 1 } end
		if base == pc then tries[#tries + 1] = { letter, 0, 2 } end
		for _, acc in ipairs({ 1, -1 }) do
			if (base + acc) % 12 == pc then
				local leading = key.minor and acc == 1 and pc == (key.pc + 11) % 12
				tries[#tries + 1] = { letter, acc, leading and 3 or ((acc == 1) == (key.sharps >= 0)) and 4 or 5 }
			end
		end
	end
	table.sort(tries, function(a, b) return a[3] < b[3] end)
	local letter, acc = tries[1][1], tries[1][2]
	return letter, acc, (m - acc - LETTER_PC[letter]) // 12 - 1
end

local function chord_root(name, shift, key)
	local pc = (name_pc(name) + shift) % 12
	for letter, base in pairs(LETTER_PC) do
		if (base + key.sig[letter]) % 12 == pc then
			return letter .. (key.sig[letter] == 1 and "#" or key.sig[letter] == -1 and "b" or "")
		end
	end
	return (key.sharps >= 0 and SHARP_NAMES or FLAT_NAMES)[pc]
end

-- Walks one body line. With `to` it writes the transposed line; `pitches`
-- collects every MIDI pitch read (from `from`'s point of view), for the check.
local function walk(line, from, to, shift, pitches)
	local out, i = {}, 1
	local seen, written = {}, {}          -- accidentals in force in this bar, by letter .. octave
	while i <= #line do
		local c = line:sub(i, i)
		local acc_text, letter, marks = line:match("^([_^=]*)([A-Ga-g])([,']*)", i)
		if c == '"' then
			local close = line:find('"', i + 1, true) or #line
			local chord = line:sub(i + 1, close - 1)
			local root, suffix, bass = chord:match("^([A-G][#b]?)([^/]*)/?([A-G]?[#b]?)$")
			if to and root then
				chord = chord_root(root, shift, to) .. suffix .. (bass ~= "" and "/" .. chord_root(bass, shift, to) or "")
			end
			out[#out + 1] = '"' .. chord .. '"'
			i = close + 1
		elseif letter then
			local up = letter:upper()
			local octave = (letter == up and 4 or 5) + select(2, marks:gsub("'", "")) - select(2, marks:gsub(",", ""))
			local slot = up .. octave
			if acc_text ~= "" then seen[slot] = assert(ACC_VALUE[acc_text], "accidental '" .. acc_text .. "'") end
			local m = LETTER_PC[up] + (seen[slot] or from.sig[up]) + 12 * (octave + 1)
			pitches[#pitches + 1] = m
			if to then
				local l, acc, o = spell(m + shift, to)
				local text = ""
				if (written[l .. o] or to.sig[l]) ~= acc then
					text = ACC_TEXT[acc]
					written[l .. o] = acc
				end
				out[#out + 1] = text .. (o >= 5 and l:lower() .. ("'"):rep(o - 5) or l .. (","):rep(4 - o))
			end
			i = i + #acc_text + 1 + #marks
		else
			if c == "|" then seen, written = {}, {} end
			out[#out + 1] = c
			i = i + 1
		end
	end
	return table.concat(out)
end

-- Every body line of `abc` through fn(line, voice); header fields and comments pass.
local function each_line(abc, fn)
	local out, voice = {}, nil
	for text in (abc .. "\n"):gmatch("(.-)\n") do
		local line = text
		if line:match("^V:") then
			voice = line:match("^V:%s*(%S+)")
		elseif not line:match("^%a:") and not line:match("^%%") then
			line = fn(line, voice)
		end
		out[#out + 1] = line
	end
	return table.concat(out, "\n")
end

local function transpose(abc, semitones, vocal_octaves)
	local field = assert(abc:match("\n(K:[^\n]*)"), "score has no K: field")
	local from = parse_key(field)
	local to = make_key((from.pc + semitones) % 12, from.minor)
	local function shift_of(voice) return semitones + (voice == "Vocal" and 12 * vocal_octaves or 0) end

	local before = {}
	local result = each_line(abc, function(line, voice)
		local at = #before
		local new = walk(line, from, to, shift_of(voice), before)
		for k = at + 1, #before do before[k] = before[k] + shift_of(voice) end
		return new
	end)
	result = result:gsub("\nK:[^\n]*", function() return "\nK:" .. to.name end, 1)

	local after = {}
	each_line(result, function(line) walk(line, to, nil, 0, after); return line end)
	assert(#after == #before, ("transpose check: %d notes became %d"):format(#before, #after))
	for k = 1, #before do
		assert(after[k] == before[k], ("transpose check: note %d is MIDI %d, wanted %d"):format(k, after[k], before[k]))
	end
	return result, from.name .. " -> " .. to.name
end

-- As a script `...` holds the command line; through dofile()/require it is empty.
local path, semitones, vocal_octaves = ...
if not path then
	return transpose
end
semitones     = tonumber(semitones)
vocal_octaves = tonumber(vocal_octaves) or 0
if not semitones then
	io.stderr:write("usage: abc_transpose.lua SCORE.abc SEMITONES [VOCAL_OCTAVES] > new.abc\n")
	os.exit(2)
end
local file = assert(io.open(path, "r"))
local abc = file:read("a")
file:close()
local result, moved = transpose(abc, semitones, vocal_octaves)
io.stderr:write(moved .. ", vocal line " .. ("%+d"):format(semitones + 12 * vocal_octaves) .. " semitones (pitch-checked)\n")
io.stdout:write(result)
