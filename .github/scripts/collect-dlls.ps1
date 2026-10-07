# Copy the DLLs an executable needs from one directory (vcpkg's bin) next to
# it, following their own imports too. System DLLs, the MSVC runtime and
# vulkan-1.dll are never in SearchDir, so they are never copied.
#
#	collect-dlls.ps1 -Exe build/yue2.exe -SearchDir <vcpkg>/installed/x64-windows/bin -Dest build
#
# Needs dumpbin on PATH (the MSVC developer environment).

param(
	[Parameter(Mandatory = $true)] [string] $Exe,
	[Parameter(Mandatory = $true)] [string] $SearchDir,
	[Parameter(Mandatory = $true)] [string] $Dest
)

$ErrorActionPreference = 'Stop'

$seen  = @{}
$queue = [System.Collections.Generic.Queue[string]]::new()
$queue.Enqueue((Resolve-Path $Exe).Path)
while ($queue.Count -gt 0)
{
	$file = $queue.Dequeue()
	foreach ($line in (dumpbin /nologo /dependents $file))
	{
		$name = $line.Trim()
		if ($name -notmatch '\.dll$' -or $seen.ContainsKey($name.ToLower()))
		{
			continue
		}
		$seen[$name.ToLower()] = $true
		$src = Join-Path $SearchDir $name
		if (Test-Path $src)
		{
			Copy-Item $src $Dest -Force
			Write-Host "copied  $name"
			$queue.Enqueue($src)
		} else {
			Write-Host "system  $name"
		}
	}
}
