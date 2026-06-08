# Fix test files: remove outer #[cfg(test)] mod tests { ... } wrapper
# because #[path] declaration already creates the tests module.
# Then unindent contents by 4 spaces.

$testsDir = "C:\Users\John\Desktop\AI Gens\Rust\SpynalTap\lib\tests\unit"
$fixed = 0
$skipped = 0

Get-ChildItem -Path $testsDir -Recurse -Filter "*.rs" | ForEach-Object {
    $file = $_.FullName
    $lines = [System.IO.File]::ReadAllLines($file)
    
    # Check if file starts with #[cfg(test)] then mod tests {
    if ($lines.Count -lt 3) { $skipped++; return }
    if ($lines[0].Trim() -ne '#[cfg(test)]') { $skipped++; return }
    if ($lines[1].Trim() -notmatch '^mod \w+\s*\{\s*$') { $skipped++; return }
    
    # Find the matching closing brace
    $braceCount = 0
    $lastLine = -1
    for ($i = 1; $i -lt $lines.Count; $i++) {
        foreach ($c in $lines[$i].ToCharArray()) {
            if ($c -eq '{') { $braceCount++ }
            if ($c -eq '}') { $braceCount-- }
        }
        if ($braceCount -eq 0) {
            $lastLine = $i
            break
        }
    }
    
    if ($lastLine -le 2) { $skipped++; return }
    
    # Extract inner content (lines 2..lastLine-1), unindent by 4 spaces
    $inner = @()
    for ($i = 2; $i -lt $lastLine; $i++) {
        $line = $lines[$i]
        if ($line.Length -ge 4 -and $line.Substring(0,4) -eq '    ') {
            $line = $line.Substring(4)
        } elseif ($line.Length -gt 0 -and $line.Trim().Length -eq 0) {
            $line = ''
        }
        $inner += $line
    }
    
    # Write back
    $content = $inner -join "`r`n"
    [System.IO.File]::WriteAllText($file, $content + "`r`n", [System.Text.UTF8Encoding]::new($false))
    Write-Host "  Fixed: $($file.Substring($testsDir.Length))"
    $fixed++
}

Write-Host "Fixed $fixed files, skipped $skipped"
