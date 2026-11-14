Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path Env:CONDA_PREFIX)) {
    throw "Activate the sphinx_env conda environment before building RAG documentation."
}
$activeEnvironment = Split-Path -Leaf $env:CONDA_PREFIX
if ($activeEnvironment -ne "sphinx_env") {
    throw "Activate the sphinx_env conda environment before building RAG documentation."
}

$python = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found in the active sphinx_env environment."
}
$relicPython = Join-Path (
    Split-Path -Parent $env:CONDA_PREFIX
) "RELIC\python.exe"
if (-not (Test-Path -LiteralPath $relicPython)) {
    throw "The sibling RELIC environment is required for the live science inventory check."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $PSScriptRoot "source\public"
$htmlRoot = Join-Path $PSScriptRoot "build"
$ragRoot = Join-Path $PSScriptRoot "rag"
$validator = Join-Path $PSScriptRoot "validate_rag_docs.py"
$scienceGenerator = Join-Path $PSScriptRoot "generate_public_science_reference.py"
$tempRoot = Join-Path (
    [System.IO.Path]::GetTempPath()
) ("relic-rag-docs-" + [guid]::NewGuid().ToString("N"))
$stageRoot = Join-Path $tempRoot "publish"
$compatRoot = Join-Path $tempRoot "compat"

function Convert-ToPortableRagMarkdown {
    param([Parameter(Mandatory = $true)][string]$Path)

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $content = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    $content = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        '!\[([^\]]*)\]\([^)]+\)',
        '$1'
    )
    $content = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        '(?<=\]\()[^)\r\n]+(?=\))',
        {
            param($match)
            $match.Value.Replace([char]92, [char]47)
        }
    )
    if ([System.IO.Path]::GetFileName($Path) -eq "llms.txt") {
        $content = [System.Text.RegularExpressions.Regex]::Replace(
            $content,
            '(?m)^(- \[[^\r\n]+?\]\([^)]+\): )(.*)$',
            {
                param($match)
                $description = [System.Text.RegularExpressions.Regex]::Replace(
                    $match.Groups[2].Value,
                    '\[([^\]]+)\]\([^)]+\)',
                    '$1'
                )
                $match.Groups[1].Value + $description
            }
        )
    }
    if (
        [System.Text.RegularExpressions.Regex]::IsMatch(
            $content,
            '(?m)(?:\t| {2,})(?=\r?$)'
        )
    ) {
        throw "RAG output contains a Markdown hard break or trailing tab: $Path"
    }
    $content = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        '(?m) (?=\r?$)',
        ''
    )
    [System.IO.File]::WriteAllText($Path, $content, $utf8NoBom)
}

function Test-MarkdownPipeRow {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line
    )

    $trimmed = $Line.Trim()
    return (
        $trimmed.Length -ge 2 -and
        $trimmed.StartsWith("|") -and
        $trimmed.EndsWith("|")
    )
}

function Get-MarkdownPipeCells {
    param([Parameter(Mandatory = $true)][string]$Line)

    if (-not (Test-MarkdownPipeRow -Line $Line)) {
        throw "Expected a Markdown pipe-table row."
    }
    $trimmed = $Line.Trim()
    $cells = [System.Collections.Generic.List[string]]::new()
    $builder = [System.Text.StringBuilder]::new()
    $backslashRun = 0
    for ($index = 1; $index -lt $trimmed.Length - 1; $index++) {
        $character = $trimmed[$index]
        if ($character -eq [char]124 -and ($backslashRun % 2) -eq 0) {
            $cells.Add($builder.ToString())
            [void]$builder.Clear()
            $backslashRun = 0
            continue
        }
        [void]$builder.Append($character)
        if ($character -eq [char]92) {
            $backslashRun += 1
        } else {
            $backslashRun = 0
        }
    }
    $cells.Add($builder.ToString())
    return $cells.ToArray()
}

function Test-MarkdownSeparatorRow {
    param([Parameter(Mandatory = $true)][string[]]$Cells)

    if ($Cells.Count -eq 0) {
        return $false
    }
    foreach ($cell in $Cells) {
        if ($cell -notmatch '^\s*:?-{3,}:?\s*$') {
            return $false
        }
    }
    return $true
}

function Convert-MarkdownTableRowPipes {
    param(
        [Parameter(Mandatory = $true)][string]$Line,
        [Parameter(Mandatory = $true)][int]$ExpectedWidth,
        [Parameter(Mandatory = $true)][string]$Source
    )

    $lineEndingMatch = [System.Text.RegularExpressions.Regex]::Match(
        $Line,
        '(\r\n|\n|\r)$'
    )
    $lineEnding = $lineEndingMatch.Value
    $bodyLength = $Line.Length - $lineEnding.Length
    $body = $Line.Substring(0, $bodyLength)
    $rawPipeCount = ($body.ToCharArray() | Where-Object { $_ -eq [char]124 }).Count

    $converted = [System.Text.RegularExpressions.Regex]::Replace(
        $body,
        '<code(?<attributes>[^>]*)>(?<value>[^<\r\n]*\|[^<\r\n]*)</code>',
        {
            param($match)
            $value = $match.Groups["value"].Value.Replace("|", "&#124;")
            "<code$($match.Groups["attributes"].Value)>$value</code>"
        },
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    $rightTrimmed = $converted.TrimEnd()
    $trailingWhitespace = $converted.Substring($rightTrimmed.Length)
    $firstPipe = $rightTrimmed.IndexOf([char]124)
    $lastPipe = $rightTrimmed.LastIndexOf([char]124)
    $builder = [System.Text.StringBuilder]::new()
    for ($index = 0; $index -lt $rightTrimmed.Length; $index++) {
        $character = $rightTrimmed[$index]
        if (
            $character -eq [char]124 -and
            $index -ne $firstPipe -and
            $index -ne $lastPipe
        ) {
            $previousIsSpace = [char]::IsWhiteSpace($rightTrimmed[$index - 1])
            $nextIsSpace = [char]::IsWhiteSpace($rightTrimmed[$index + 1])
            if (-not ($previousIsSpace -and $nextIsSpace)) {
                [void]$builder.Append("&#124;")
                continue
            }
        }
        [void]$builder.Append($character)
    }
    $converted = $builder.ToString() + $trailingWhitespace
    $cells = @(Get-MarkdownPipeCells -Line $converted)
    if ($cells.Count -ne $ExpectedWidth) {
        throw (
            "RAG table row width mismatch after pipe normalization in " +
            "$Source; expected=$ExpectedWidth actual=$($cells.Count): $body"
        )
    }
    $remainingPipeCount = (
        $converted.ToCharArray() |
            Where-Object { $_ -eq [char]124 }
    ).Count
    return [pscustomobject]@{
        Text = $converted + $lineEnding
        EncodedPipes = $rawPipeCount - $remainingPipeCount
    }
}

function Convert-MarkdownTableCellPipes {
    param([Parameter(Mandatory = $true)][string]$Path)

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $content = [System.IO.File]::ReadAllText(
        $Path,
        [System.Text.Encoding]::UTF8
    )
    $segments = [System.Text.RegularExpressions.Regex]::Split(
        $content,
        '(?<=\n)'
    )
    $tableCount = 0
    $rowCount = 0
    $cellCount = 0
    $changedLineCount = 0
    $encodedPipeCount = 0
    for ($index = 0; $index -lt $segments.Count - 1; $index++) {
        $header = $segments[$index].TrimEnd([char]13, [char]10)
        $separator = $segments[$index + 1].TrimEnd([char]13, [char]10)
        if (
            -not (Test-MarkdownPipeRow -Line $header) -or
            -not (Test-MarkdownPipeRow -Line $separator)
        ) {
            continue
        }
        $headerCells = @(Get-MarkdownPipeCells -Line $header)
        $separatorCells = @(Get-MarkdownPipeCells -Line $separator)
        if (-not (Test-MarkdownSeparatorRow -Cells $separatorCells)) {
            continue
        }
        if ($headerCells.Count -ne $separatorCells.Count) {
            throw (
                "RAG table header width mismatch in $Path; " +
                "header=$($headerCells.Count) separator=$($separatorCells.Count)"
            )
        }

        $expectedWidth = $separatorCells.Count
        $tableCount += 1
        $rowIndex = $index
        while ($rowIndex -lt $segments.Count) {
            $row = $segments[$rowIndex].TrimEnd([char]13, [char]10)
            if (-not (Test-MarkdownPipeRow -Line $row)) {
                break
            }
            if ($rowIndex -ne $index + 1) {
                $result = Convert-MarkdownTableRowPipes `
                    -Line $segments[$rowIndex] `
                    -ExpectedWidth $expectedWidth `
                    -Source $Path
                if ($result.Text -ne $segments[$rowIndex]) {
                    $changedLineCount += 1
                    $segments[$rowIndex] = $result.Text
                }
                $encodedPipeCount += $result.EncodedPipes
                $rowCount += 1
                $cellCount += $expectedWidth
            }
            $rowIndex += 1
        }
        $index = $rowIndex - 1
    }
    $normalized = [string]::Concat($segments)
    if ($normalized -ne $content) {
        [System.IO.File]::WriteAllText($Path, $normalized, $utf8NoBom)
    }
    return [pscustomobject]@{
        Tables = $tableCount
        Rows = $rowCount
        Cells = $cellCount
        ChangedLines = $changedLineCount
        EncodedPipes = $encodedPipeCount
    }
}

function Convert-LinksForCorpusRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Content,
        [Parameter(Mandatory = $true)][string]$PagePath,
        [Parameter(Mandatory = $true)][string]$LanguageRoot
    )

    $languageFullPath = [System.IO.Path]::GetFullPath($LanguageRoot)
    $languagePrefix = $languageFullPath.TrimEnd([char]92) + [char]92
    $pageDirectory = Split-Path -Parent $PagePath
    return [System.Text.RegularExpressions.Regex]::Replace(
        $Content,
        '(?<=\]\()[^)\r\n]+(?=\))',
        {
            param($match)
            $rawTarget = $match.Value
            $wrapped = (
                $rawTarget.StartsWith("<") -and
                $rawTarget.EndsWith(">")
            )
            $target = if ($wrapped) {
                $rawTarget.Substring(1, $rawTarget.Length - 2)
            } else {
                $rawTarget
            }
            if (
                $target.StartsWith("#") -or
                $target -match '^[A-Za-z][A-Za-z0-9+.-]*:'
            ) {
                return $rawTarget
            }

            $separatorIndex = $target.Length
            foreach ($separator in @("#", "?")) {
                $candidateIndex = $target.IndexOf($separator)
                if (
                    $candidateIndex -ge 0 -and
                    $candidateIndex -lt $separatorIndex
                ) {
                    $separatorIndex = $candidateIndex
                }
            }
            $pathPart = $target.Substring(0, $separatorIndex)
            $suffix = $target.Substring($separatorIndex)
            if ([string]::IsNullOrWhiteSpace($pathPart)) {
                return $rawTarget
            }

            $candidate = if ($pathPart.StartsWith("/")) {
                Join-Path $languageFullPath $pathPart.TrimStart([char]47)
            } else {
                Join-Path $pageDirectory $pathPart
            }
            $candidate = [System.IO.Path]::GetFullPath($candidate)
            if (
                $candidate -ne $languageFullPath -and
                -not $candidate.StartsWith(
                    $languagePrefix,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            ) {
                throw "RAG link escapes the language root: $PagePath -> $target"
            }
            $relativeTarget = $candidate.Substring(
                $languageFullPath.Length
            ).TrimStart([char]92).Replace([char]92, [char]47)
            $portableTarget = $relativeTarget + $suffix
            if ($wrapped) {
                return "<$portableTarget>"
            }
            return $portableTarget
        }
    )
}

function Build-PortableLlmsFull {
    param([Parameter(Mandatory = $true)][string]$LanguageRoot)

    $languageFullPath = [System.IO.Path]::GetFullPath($LanguageRoot)
    $pageRecords = @(
        Get-ChildItem `
            -LiteralPath $languageFullPath `
            -Filter "*.html.md" `
            -File `
            -Recurse |
            ForEach-Object {
                [pscustomobject]@{
                    Path = $_.FullName
                    Relative = $_.FullName.Substring(
                        $languageFullPath.Length
                    ).TrimStart([char]92).Replace([char]92, [char]47)
                }
            }
    )
    $pageRecords = @(
        $pageRecords |
            Sort-Object `
                @{ Expression = {
                    if ($_.Relative -eq "index.html.md") { 0 } else { 1 }
                } }, `
                @{ Expression = { $_.Relative } }
    )

    $builder = [System.Text.StringBuilder]::new()
    foreach ($record in $pageRecords) {
        $pageContent = [System.IO.File]::ReadAllText(
            $record.Path,
            [System.Text.Encoding]::UTF8
        ).Trim()
        $portableContent = Convert-LinksForCorpusRoot `
            -Content $pageContent `
            -PagePath $record.Path `
            -LanguageRoot $languageFullPath
        [void]$builder.Append("# ")
        [void]$builder.AppendLine($record.Relative)
        [void]$builder.AppendLine()
        [void]$builder.AppendLine($portableContent)
        [void]$builder.AppendLine()
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $languageFullPath "llms-full.txt"),
        $builder.ToString(),
        [System.Text.UTF8Encoding]::new($false)
    )
}

$resolvedRag = [System.IO.Path]::GetFullPath($ragRoot)
$expectedRag = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "docs\rag"))
if ($resolvedRag -ne $expectedRag) {
    throw "Resolved RAG target does not match docs\rag."
}

& $python -c @"
from importlib.metadata import version

expected = {
    'sphinx-llm': '0.4.1',
    'sphinx-markdown-builder': '0.6.10',
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f'RAG dependency mismatch: expected={expected}, actual={actual}')
print('[OK] RAG dependencies:', actual)
"@
if ($LASTEXITCODE -ne 0) {
    throw "Install docs\requirements-rag.txt in the active sphinx_env environment."
}

if (-not (Test-Path -LiteralPath (Join-Path $htmlRoot "en\index.html"))) {
    throw "Build docs\build before generating the RAG corpus."
}
if (-not (Test-Path -LiteralPath (Join-Path $htmlRoot "zh\index.html"))) {
    throw "Build docs\build before generating the RAG corpus."
}

& $relicPython $scienceGenerator --check
if ($LASTEXITCODE -ne 0) {
    throw "The generated scientific class reference is stale or incomplete."
}

New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
New-Item -ItemType Directory -Force -Path $compatRoot | Out-Null
$relicSitePackages = Join-Path (
    Split-Path -Parent $relicPython
) "Lib\site-packages"
if (-not (Test-Path -LiteralPath $relicSitePackages)) {
    throw "RELIC site-packages were not found for the documentation parser fallback."
}
$siteCustomize = Join-Path $compatRoot "sitecustomize.py"
$escapedRelicSitePackages = $relicSitePackages.Replace("'", "''")
$siteCustomizeContent = @"
import sys

# Keep sphinx_env authoritative and append only missing RELIC parser dependencies.
sys.path.append(r'$escapedRelicSitePackages')
"@
[System.IO.File]::WriteAllText(
    $siteCustomize,
    $siteCustomizeContent,
    [System.Text.UTF8Encoding]::new($false)
)

$previousEnable = $env:RELIC_ENABLE_SPHINX_LLM
$previousLanguage = $env:RELIC_DOC_LANGUAGE
$previousPythonPath = $env:PYTHONPATH

try {
    $env:RELIC_ENABLE_SPHINX_LLM = "1"
    $env:PYTHONPATH = $compatRoot
    foreach ($language in @("en", "zh_CN")) {
        $publicLanguage = if ($language -eq "zh_CN") { "zh" } else { "en" }
        $env:RELIC_DOC_LANGUAGE = $language
        $outputPath = Join-Path $tempRoot ("html-" + $publicLanguage)
        $doctreePath = Join-Path $tempRoot ("doctrees-" + $publicLanguage)
        & $python -m sphinx `
            -E -a -W --keep-going `
            -D "language=$language" `
            -d $doctreePath `
            -b html `
            $sourceRoot `
            $outputPath
        if ($LASTEXITCODE -ne 0) {
            throw "sphinx-llm build failed for $language."
        }

        $destination = Join-Path $stageRoot $publicLanguage
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        foreach ($name in @("llms.txt", "llms-full.txt")) {
            $source = Join-Path $outputPath $name
            if (-not (Test-Path -LiteralPath $source)) {
                throw "sphinx-llm did not generate $source."
            }
            Copy-Item -LiteralPath $source -Destination $destination
        }
        Get-ChildItem -LiteralPath $outputPath -Filter "*.html.md" -File -Recurse |
            ForEach-Object {
                $relativePath = $_.FullName.Substring(
                    $outputPath.Length
                ).TrimStart("\")
                $targetPath = Join-Path $destination $relativePath
                New-Item `
                    -ItemType Directory `
                    -Force `
                    -Path (Split-Path -Parent $targetPath) |
                    Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $targetPath
            }
    }
} finally {
    if ($null -eq $previousEnable) {
        Microsoft.PowerShell.Management\Remove-Item Env:RELIC_ENABLE_SPHINX_LLM -ErrorAction SilentlyContinue
    } else {
        $env:RELIC_ENABLE_SPHINX_LLM = $previousEnable
    }
    if ($null -eq $previousLanguage) {
        Microsoft.PowerShell.Management\Remove-Item Env:RELIC_DOC_LANGUAGE -ErrorAction SilentlyContinue
    } else {
        $env:RELIC_DOC_LANGUAGE = $previousLanguage
    }
    if ($null -eq $previousPythonPath) {
        Microsoft.PowerShell.Management\Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

Get-ChildItem -LiteralPath $stageRoot -File -Recurse |
    ForEach-Object {
        Convert-ToPortableRagMarkdown -Path $_.FullName
    }
$tableTotals = [pscustomobject]@{
    Tables = 0
    Rows = 0
    Cells = 0
    ChangedLines = 0
    EncodedPipes = 0
}
Get-ChildItem -LiteralPath $stageRoot -Filter "*.html.md" -File -Recurse |
    ForEach-Object {
        $stats = Convert-MarkdownTableCellPipes -Path $_.FullName
        $tableTotals.Tables += $stats.Tables
        $tableTotals.Rows += $stats.Rows
        $tableTotals.Cells += $stats.Cells
        $tableTotals.ChangedLines += $stats.ChangedLines
        $tableTotals.EncodedPipes += $stats.EncodedPipes
    }
Write-Host (
    "[OK] Normalized RAG tables: tables=$($tableTotals.Tables) " +
    "rows=$($tableTotals.Rows) cells=$($tableTotals.Cells) " +
    "changed_lines=$($tableTotals.ChangedLines) " +
    "encoded_pipes=$($tableTotals.EncodedPipes)"
)
foreach ($language in @("en", "zh")) {
    Build-PortableLlmsFull -LanguageRoot (Join-Path $stageRoot $language)
}

& $python $validator $htmlRoot $stageRoot
if ($LASTEXITCODE -ne 0) {
    throw "RAG corpus validation failed."
}

if (-not (Test-Path -LiteralPath $ragRoot)) {
    New-Item -ItemType Directory -Force -Path $ragRoot | Out-Null
}
foreach ($language in @("en", "zh")) {
    $sourceLanguage = Join-Path $stageRoot $language
    $destinationLanguage = Join-Path $ragRoot $language
    New-Item -ItemType Directory -Force -Path $destinationLanguage | Out-Null
    & robocopy.exe `
        $sourceLanguage `
        $destinationLanguage `
        /MIR `
        /COPY:DAT `
        /DCOPY:DAT `
        /R:2 `
        /W:1 `
        /NFL `
        /NDL `
        /NJH `
        /NJS `
        /NP
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -ge 8) {
        throw (
            "Publishing docs\rag\$language failed with robocopy exit code " +
            "$robocopyExitCode."
        )
    }
}

$resolvedTempRoot = [System.IO.Path]::GetFullPath($tempRoot)
$resolvedSystemTemp = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::GetTempPath()
)
if (-not $resolvedTempRoot.StartsWith(
    $resolvedSystemTemp,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Refusing to remove a RAG temporary path outside the system temp directory."
}
Microsoft.PowerShell.Management\Remove-Item -LiteralPath $resolvedTempRoot -Recurse -Force

Write-Host "[OK] RELIC RAG corpus built with sphinx-llm at $ragRoot"

# Robocopy uses 1-7 for successful publication with changes.
# All native command failures have already been checked above.
exit 0
