Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path Env:CONDA_PREFIX)) {
    throw "Activate the sphinx_env conda environment before building documentation."
}
$activeEnvironment = Split-Path -Leaf $env:CONDA_PREFIX
if ($activeEnvironment -ne "sphinx_env") {
    throw "Activate the sphinx_env conda environment before building documentation."
}

$python = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found in the active sphinx_env environment."
}
$relicPython = Join-Path (
    Split-Path -Parent $env:CONDA_PREFIX
) "RELIC\python.exe"
if (-not (Test-Path -LiteralPath $relicPython)) {
    throw "The sibling RELIC environment is required for live documentation checks."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $PSScriptRoot "source\public"
$buildRoot = Join-Path $PSScriptRoot "build"
$scienceGenerator = Join-Path $PSScriptRoot "generate_public_science_reference.py"
$htmlValidator = Join-Path $PSScriptRoot "validate_public_html.py"
$tempRoot = Join-Path (
    [System.IO.Path]::GetTempPath()
) ("relic-public-docs-" + [guid]::NewGuid().ToString("N"))
$stageRoot = Join-Path $tempRoot "publish"

function Invoke-SphinxBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Language,
        [Parameter(Mandatory = $true)]
        [string]$Builder,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $doctreePath = Join-Path $tempRoot ("doctrees-" + $Language + "-" + $Builder)
    & $python -m sphinx `
        -E -a -W --keep-going `
        -D "language=$Language" `
        -d $doctreePath `
        -b $Builder `
        $sourceRoot `
        $OutputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Sphinx $Builder build failed for $Language."
    }
}

function Copy-PublicHtml {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
    Get-ChildItem -LiteralPath $SourcePath -Filter "*.html" -File -Recurse |
        ForEach-Object {
            $relativePath = $_.FullName.Substring($SourcePath.Length).TrimStart("\")
            $targetPath = Join-Path $DestinationPath $relativePath
            New-Item `
                -ItemType Directory `
                -Force `
                -Path (Split-Path -Parent $targetPath) |
                Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $targetPath
        }
    Copy-Item `
        -LiteralPath (Join-Path $SourcePath "searchindex.js") `
        -Destination $DestinationPath
    Copy-Item `
        -LiteralPath (Join-Path $SourcePath "_static") `
        -Destination $DestinationPath `
        -Recurse
    Copy-Item `
        -LiteralPath (Join-Path $SourcePath "_images") `
        -Destination $DestinationPath `
        -Recurse
}

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

& $relicPython $scienceGenerator --check
if ($LASTEXITCODE -ne 0) {
    throw "The generated scientific class reference is stale or incomplete."
}

$htmlEn = Join-Path $tempRoot "html-en"
$htmlZh = Join-Path $tempRoot "html-zh"

Invoke-SphinxBuild -Language "en" -Builder "html" -OutputPath $htmlEn
Invoke-SphinxBuild -Language "zh_CN" -Builder "html" -OutputPath $htmlZh
Invoke-SphinxBuild `
    -Language "en" `
    -Builder "linkcheck" `
    -OutputPath (Join-Path $tempRoot "linkcheck-en")
Invoke-SphinxBuild `
    -Language "zh_CN" `
    -Builder "linkcheck" `
    -OutputPath (Join-Path $tempRoot "linkcheck-zh")

$compiledLocaleSource = Join-Path $PSScriptRoot "locale\zh_CN\LC_MESSAGES"
$compiledLocaleTemp = Join-Path $tempRoot "compiled-locale"
New-Item -ItemType Directory -Force -Path $compiledLocaleTemp | Out-Null
Get-ChildItem -LiteralPath $compiledLocaleSource -Filter "*.mo" -File -Recurse |
    ForEach-Object {
        $relativePath = $_.FullName.Substring(
            $compiledLocaleSource.Length
        ).TrimStart("\")
        $targetPath = Join-Path $compiledLocaleTemp $relativePath
        New-Item `
            -ItemType Directory `
            -Force `
            -Path (Split-Path -Parent $targetPath) |
            Out-Null
        Move-Item -LiteralPath $_.FullName -Destination $targetPath
    }

Copy-Item `
    -LiteralPath (Join-Path $sourceRoot "language_index.html") `
    -Destination (Join-Path $stageRoot "index.html")
$projectVersion = (
    & $relicPython -c `
        "import sys, tomllib, pathlib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))['project']['version'])" `
        (Join-Path $repoRoot "pyproject.toml")
).Trim()
if ($LASTEXITCODE -ne 0 -or -not $projectVersion) {
    throw "Unable to resolve the RELIC package version."
}
$landingPath = Join-Path $stageRoot "index.html"
$landingText = [System.IO.File]::ReadAllText($landingPath)
$landingText = $landingText.Replace("{{RELIC_VERSION}}", $projectVersion)
[System.IO.File]::WriteAllText(
    $landingPath,
    $landingText,
    [System.Text.UTF8Encoding]::new($false)
)
Copy-PublicHtml -SourcePath $htmlEn -DestinationPath (Join-Path $stageRoot "en")
Copy-PublicHtml -SourcePath $htmlZh -DestinationPath (Join-Path $stageRoot "zh")

$forbiddenNames = @(
    ".buildinfo",
    "objects.inv",
    "genindex.html"
)
$forbiddenExtensions = @(
    ".doctree",
    ".pickle",
    ".log",
    ".aux",
    ".toc",
    ".txt"
)

foreach ($item in Get-ChildItem -LiteralPath $stageRoot -Recurse -Force) {
    if ($forbiddenNames -contains $item.Name) {
        throw "Forbidden publication artifact: $($item.FullName)"
    }
    if ($item.PSIsContainer -and $item.Name -in @("_sources", ".doctrees", "linkcheck", "test")) {
        throw "Forbidden publication directory: $($item.FullName)"
    }
    if (-not $item.PSIsContainer -and $forbiddenExtensions -contains $item.Extension) {
        throw "Forbidden publication file: $($item.FullName)"
    }
}

$expectedTopLevel = @("en", "index.html", "zh")
$actualTopLevel = @(
    Get-ChildItem -LiteralPath $stageRoot -Force |
        Sort-Object Name |
        ForEach-Object Name
)
if (Compare-Object $expectedTopLevel $actualTopLevel) {
    throw "Unexpected top-level publication content."
}

$enPages = @(
    $languageRoot = Join-Path $stageRoot "en"
    Get-ChildItem -LiteralPath $languageRoot -Filter "*.html" -File -Recurse |
        ForEach-Object {
            $_.FullName.Substring($languageRoot.Length).TrimStart("\")
        } |
        Sort-Object
)
$zhPages = @(
    $languageRoot = Join-Path $stageRoot "zh"
    Get-ChildItem -LiteralPath $languageRoot -Filter "*.html" -File -Recurse |
        ForEach-Object {
            $_.FullName.Substring($languageRoot.Length).TrimStart("\")
        } |
        Sort-Object
)
if (Compare-Object $enPages $zhPages) {
    throw "English and Chinese page sets differ."
}

$expectedPages = @(
    "api\core_runtime.html",
    "api\index.html",
    "api\orchestration_validation.html",
    "api\parameters_contracts.html",
    "api\scientific_structure_extensions.html",
    "architecture.html",
    "extension_contracts.html",
    "index.html",
    "operations.html",
    "orchestration.html",
    "science\anthroposphere.html",
    "science\atmosphere.html",
    "science\biosphere.html",
    "science\hydrosphere.html",
    "science\index.html",
    "science\isotopes.html",
    "science\pedosphere.html",
    "science\transport.html",
    "search.html",
    "theory\anthroposphere.html",
    "theory\atmosphere.html",
    "theory\biosphere.html",
    "theory\dec_iphs_generic.html",
    "theory\foundations.html",
    "theory\hydrosphere.html",
    "theory\index.html",
    "theory\isotopes_transport.html",
    "theory\numerics.html",
    "theory\pedosphere.html",
    "workflows\calibration_sensitivity_and_uq.html",
    "workflows\cli_and_extensions.html",
    "workflows\configuration_and_parameters.html",
    "workflows\coupling_workflows.html",
    "workflows\devices_and_distributed.html",
    "workflows\graph_validation_and_execution.html",
    "workflows\index.html",
    "workflows\installation_and_imports.html",
    "workflows\persistence_outputs_and_data_io.html",
    "workflows\state_functors_and_simulation.html"
)
if (Compare-Object $expectedPages $enPages) {
    throw "Unexpected English publication page set."
}

$bannedBodyPattern = "pytest|test[\\/]|sphinx-build|doctree|linkcheck|phase8_scientific_iphs_standardization_plan|typed_cochain_morphism_enhancement_plan"
$contentPages = @($enPages | Where-Object { $_ -ne "search.html" })
foreach ($page in $contentPages) {
    $enText = [System.IO.File]::ReadAllText((Join-Path $stageRoot "en\$page"))
    $zhText = [System.IO.File]::ReadAllText((Join-Path $stageRoot "zh\$page"))
    if ($enText -match "[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]") {
        throw "Non-English script found in en\$page."
    }
    if ($zhText -notmatch "[\u3400-\u9fff]") {
        throw "Chinese text was not found in zh\$page."
    }
    if ($enText -match $bannedBodyPattern -or $zhText -match $bannedBodyPattern) {
        throw "Internal build or development content found in $page."
    }
}

& $relicPython $htmlValidator $stageRoot
if ($LASTEXITCODE -ne 0) {
    throw "Bilingual HTML structure validation failed."
}

$resolvedBuild = [System.IO.Path]::GetFullPath($buildRoot)
$expectedBuild = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "docs\build"))
if ($resolvedBuild -ne $expectedBuild) {
    throw "Resolved build target does not match docs\build."
}

if (Test-Path -LiteralPath $buildRoot) {
    $backupRoot = Join-Path $tempRoot "previous-build"
    Copy-Item -LiteralPath $buildRoot -Destination $backupRoot -Recurse
} else {
    New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
}

& robocopy.exe `
    $stageRoot `
    $buildRoot `
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
    throw "Publishing docs\build failed with robocopy exit code $robocopyExitCode."
}

$resolvedTempRoot = [System.IO.Path]::GetFullPath($tempRoot)
$resolvedSystemTemp = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::GetTempPath()
)
if (-not $resolvedTempRoot.StartsWith(
    $resolvedSystemTemp,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Refusing to remove a documentation temporary path outside the system temp directory."
}
Microsoft.PowerShell.Management\Remove-Item `
    -LiteralPath $resolvedTempRoot `
    -Recurse `
    -Force

Write-Host "[OK] RELIC public documentation built at $buildRoot"
Write-Host "[OK] Documentation temporary files removed."

# Robocopy uses 1-7 for successful publication with changes.
# All native command failures have already been checked above.
exit 0
