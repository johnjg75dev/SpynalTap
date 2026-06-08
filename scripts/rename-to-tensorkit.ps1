# Bulk rename tensorkit/TensorKit/tensorkit → TensorKit/tensorkit across all source files
$root = "C:\Users\John\Desktop\AI Gens\Rust\TensorKit"

$pairs = @(
    # Exact project name substitutions
    @{ pattern = 'tensorkit'; replacement = 'tensorkit' }  # CLI binary name + all references
    @{ pattern = 'tensorkit';  replacement = 'tensorkit' }  # Library crate name + metadata prefix
    @{ pattern = 'TensorKit';  replacement = 'TensorKit' }  # Display name
)

# Files to process (excluding git and target)
$include = @('*.rs', '*.toml', '*.md', '*.html', '*.ps1', '*.json', '*.txt')
$exclude = @('*.lock')

Get-ChildItem -Path $root -Recurse -Include $include -Exclude $exclude |
    Where-Object { $_.FullName -notmatch '\\target\\' -and $_.FullName -notmatch '\\.git\\' } |
    ForEach-Object {
        $file = $_.FullName
        $content = [System.IO.File]::ReadAllText($file)
        $original = $content
        foreach ($p in $pairs) {
            $content = $content.Replace($p.pattern, $p.replacement)
        }
        if ($content -ne $original) {
            [System.IO.File]::WriteAllText($file, $content, [System.Text.UTF8Encoding]::new($false))
            Write-Host "  Updated: $($file.Substring($root.Length))"
        }
    }

Write-Host "`nDone. All occurrences renamed."
