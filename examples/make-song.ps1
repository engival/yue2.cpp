# Make one song. Edit the settings between the two rules in Notepad, save, then
# double-click make-song.bat (or run .\make-song.ps1 in PowerShell).
# (Linux: make-song.sh is the same thing.)
# The song lands in a folder named after $Name, next to this file:
#   $Name\$Name.flac      the song
#   $Name\lyrics.txt      the lyrics as sent
#   $Name\request.json    the request as sent
#   $Name\render\         what the model wrote (score.abc, ...) - keep it to reuse the tune
# ---------------------------------------------------------------------------

$Name = "first-song"
$Seed = 1		# another number = another song from the same words
$Gpu  = 0		# which graphics card; 0 = the first one

# Describe the sound: genre, instruments, singer. Broad common words work best.
$Style = "slow dream pop, reverb guitar, brushed drums, soft female vocal"

# [Verse], [Chorus], [Bridge], [Outro] ... on their own lines, a blank line between parts.
$Lyrics = @'
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
'@

# Extras - leave empty ("") to skip, fill in to try one.
$Tempo          = ""	# beats per minute, 20-300, e.g. 90
$Avoid          = ""	# a sound to steer away from, e.g. "minimal, repetitive, drone, ambient"
$SwitchStyle    = ""	# change to this style part-way, e.g. "90s eurodance, four-on-the-floor beat, supersaw synths, powerful female vocal"
$SwitchAtChorus = 2	# ... from this chorus on (the song is rendered twice)
$SameTuneAs     = ""	# $Name of an earlier song: sing its tune in this $Style (keep the same lyrics, pick a new $Name)

# ---------------------------------------------------------------------------

Set-Location $PSScriptRoot

$Yue2 = @("..\yue2.exe", "..\build\yue2.exe", ".\yue2.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Yue2)
{
	Write-Host "cannot find yue2.exe next to this folder"
	exit 1
}

$Request = [ordered]@{ style = $Style; lyrics_file = "lyrics.txt" }
if ($Tempo) { $Request.score_tempo = [int]$Tempo }
if ($Avoid)
{
	$Request.cfg_scale       = 3
	$Request.negative_style  = $Avoid
	$Request.negative_lyrics = $true
}
if ($SwitchStyle) { $Request.handover = @(@{ section = "chorus"; nth = [int]$SwitchAtChorus; style = $SwitchStyle }) }
if ($SameTuneAs) { $Request.abc_file = "../$SameTuneAs/render/score.abc" }
$Request.seed = [int64]$Seed

# UTF-8 without a byte-order mark, whatever version of PowerShell this is.
$Utf8 = New-Object System.Text.UTF8Encoding $false
New-Item -ItemType Directory -Force $Name | Out-Null
[System.IO.File]::WriteAllText("$PSScriptRoot\$Name\lyrics.txt", $Lyrics + "`n", $Utf8)
[System.IO.File]::WriteAllText("$PSScriptRoot\$Name\request.json", ($Request | ConvertTo-Json -Depth 5), $Utf8)

& $Yue2 song --request "$Name\request.json" --out "$Name\$Name.flac" --artifacts "$Name\render" --gpu $Gpu
if ($LASTEXITCODE -eq 0)
{
	Write-Host "done: $PSScriptRoot\$Name\$Name.flac"
}
exit $LASTEXITCODE
