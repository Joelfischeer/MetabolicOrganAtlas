# Runs the paper classification and the description generation with Claude Haiku 5.5.
# Start it from a PowerShell window in which ANTHROPIC_API_KEY is set (the script never reads or prints the key).
#
#   powershell -ExecutionPolicy Bypass -File .\run_haiku_all.ps1                   # everything, in order
#   powershell -ExecutionPolicy Bypass -File .\run_haiku_all.ps1 -Stage quick      # only the fast description steps (~15 min)
#   powershell -ExecutionPolicy Bypass -File .\run_haiku_all.ps1 -Stage classify   # only the paper classification (~2.5 h)
#   powershell -ExecutionPolicy Bypass -File .\run_haiku_all.ps1 -Stage views      # condition views + dashboard + statistics
#
# Stages (models are set in reference_network_only_metabolic/config.py: CONDITION_LLM_MODEL, DESCRIPTION_LLM_MODEL):
#   quick     organ-organ descriptions of the "All papers" view + organ descriptions + its connection types (~$1)
#   classify  LLM condition label for every paper, stops at an estimated $10 (--max-cost)   (~$6.3)
#   views     per condition view: results + connection types + organ-organ descriptions, then the dashboard + statistics (~$4)
# Every step is resumable: if it stops (PC sleep, Ctrl+C, empty credit), run the same command again.
param([ValidateSet("all", "quick", "classify", "views")] [string]$Stage = "all")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "ANTHROPIC_API_KEY is not set in this window. Set it first (see README: Using the Claude API), then run again." -ForegroundColor Red
    exit 1
}

function Run-Step([string]$Name, [string[]]$PyArgs) {
    $log = Join-Path $logDir "haiku_$Name.log"
    Write-Host "`n=== $Name ===  (log: $log)" -ForegroundColor Cyan
    & uv run python @PyArgs | ForEach-Object { $_; Add-Content -Path $log -Value $_ -Encoding utf8 }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step '$Name' failed (exit code $LASTEXITCODE). Fix the cause and run the same command again." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Backup-Dashboard {
    $html = "reference_network_only_metabolic/metabolic_literature_network.html"
    if (Test-Path $html) {
        $bak = "$html.before_haiku_$(Get-Date -Format yyyyMMdd_HHmmss).bak"
        Copy-Item $html $bak
        Write-Host "Previous dashboard saved as $bak"
    }
}

if ($Stage -in @("all", "quick")) {
    Run-Step "descriptions_all_view" @("Literature_Search/llm_descriptions.py",
        "--edge-filter", "reference_network_only_metabolic/all_organ_connections.csv",
        "--literature", "reference_network_only_metabolic/metabolic_literature_results.json",
        "--output", "reference_network_only_metabolic/metabolic_llm_descriptions.json")
    Run-Step "descriptions_organs" @("Literature_Search/organ_descriptions.py")
    # connection types of the "All papers" view (search cache is reused; this also rebuilds the dashboard)
    Backup-Dashboard
    Run-Step "types_all_view" @("run_metabolic_lit_search.py")
}

if ($Stage -in @("all", "classify")) {
    Run-Step "classify" @("run_metabolic_lit_search.py", "--classify-conditions", "--max-cost", "10")
}

if ($Stage -in @("all", "views")) {
    Run-Step "condition_views" @("run_metabolic_lit_search.py", "--condition", "every")
    Backup-Dashboard
    Run-Step "dashboard" @("run_metabolic_lit_search.py", "--viz-only")
    Run-Step "statistics" @("generate_condition_filter_statistics.py")
}

Write-Host "`nDone. Dashboard: reference_network_only_metabolic/metabolic_literature_network.html" -ForegroundColor Green
