# tools/quick_check.ps1
# Quick Check (smoke-first) with stable digests, timeouts, and non-interactive settings
# - Outputs:
#   === QUICK_CHECK_RESULT: PASS|FAIL ===
#   === QUICK_CHECK_FAIL_DIGEST: xxxxxxxxxxxxxxxx ===  (when FAIL)
#   === QUICK_CHECK_JSON_BEGIN === {...} === QUICK_CHECK_JSON_END ===
# - Fields in JSON: pass, lintFail, testFail, digest, log
# - Behavior:
#   * Non-interactive: disable pagers, colors, and pytest plugin autoload
#   * Lint: ruff (only *.py; force-exclude; ruff.toml if present; CLI fallback otherwise)
#   * Tests: pytest smoke by default; FULL_TEST=1 runs all tests
#   * Timeouts per step: pip/lint/test
#   * Stable 16-hex digest derived from step + exit + head of stderr/stdout
#   * SKIP_LINT=1 to bypass lint (emergency)

# -----------------------------
# Shell safety & encoding
# -----------------------------
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch {}

Write-Host "=== QUICK_CHECK_BEGIN ==="

# -----------------------------
# Environment (anti-stuck)
# -----------------------------
$env:PAGER = "cat"
$env:LESS = "FRX"
$env:TERM = "dumb"
$env:NO_COLOR = "1"
$env:FORCE_COLOR = "0"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PIP_NO_INPUT = "1"

# -----------------------------
# Paths & log
# -----------------------------
$ROOT = (Get-Location).Path
$LOG_DIR = Join-Path $env:TEMP "quick_check_logs"
if (!(Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }
$LOG = Join-Path $LOG_DIR ("qc_{0}.log" -f (Get-Date -Format yyyyMMdd_HHmmss))

# -----------------------------
# Timeouts (seconds)
# -----------------------------
$PIP_TIMEOUT  = 180
$LINT_TIMEOUT = 180
$TEST_TIMEOUT = 900

# -----------------------------
# Helpers
# -----------------------------
function Invoke-Proc {
  param(
    [Parameter(Mandatory=$true)][string]$File,
    [Parameter(Mandatory=$true)][string[]]$Args,
    [Parameter(Mandatory=$true)][int]$TimeoutSec,
    [Parameter(Mandatory=$true)][string]$StepName
  )
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $File
  $psi.Arguments = ($Args -join ' ')
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow  = $true

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  $null = $p.Start()

  if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    try { $p.Kill() } catch {}
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    return [pscustomobject]@{
      Step    = $StepName
      TimedOut= $true
      ExitCode= -999
      StdOut  = $stdout
      StdErr  = $stderr
    }
  } else {
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    return [pscustomobject]@{
      Step    = $StepName
      TimedOut= $false
      ExitCode= $p.ExitCode
      StdOut  = $stdout
      StdErr  = $stderr
    }
  }
}

function Make-Digest16 {
  param(
    [string]$StepName,
    [int]$ExitCode,
    [string]$StdOut,
    [string]$StdErr
  )
  $linesStdErr = if ($StdErr) { $StdErr -split "`n" } else { @() }
  $linesStdOut = if ($StdOut) { $StdOut -split "`n" } else { @() }
  $idxE = [Math]::Min(9, [Math]::Max(0, ($linesStdErr | Measure-Object).Count - 1))
  $idxO = [Math]::Min(9, [Math]::Max(0, ($linesStdOut | Measure-Object).Count - 1))
  $headStdErr = if (($linesStdErr | Measure-Object).Count -gt 0) { ($linesStdErr[0..$idxE] -join "`n") } else { "" }
  $headStdOut = if (($linesStdOut | Measure-Object).Count -gt 0) { ($linesStdOut[0..$idxO] -join "`n") } else { "" }
  $raw = "{0}|{1}|{2}|{3}" -f $StepName, $ExitCode, $headStdErr, $headStdOut
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($raw)
  $md5   = [System.Security.Cryptography.MD5]::Create()
  $hash  = ($md5.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join ''
  return $hash.Substring(0,16)
}

function Append-Log {
  param([string]$Text)
  try { Add-Content -Path $LOG -Value $Text } catch {}
}

# -----------------------------
# Main
# -----------------------------
function Invoke-QuickCheck {
  $pass = $false
  $lintFail = $false
  $testFail = $false
  $digest = ""
  $steps = @()

  # Step 1: pip install (if requirements.txt exists)
  if (Test-Path (Join-Path $ROOT "requirements.txt")) {
    $py = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $py) {
      $digest = "PYTHON_NOT_FOUND_" + (Get-Date -Format HHmmss)
      Append-Log "[pip_install] python not found"
      return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
    }
    $pipArgs = @("-m","pip","install","-q","--disable-pip-version-check","--no-input","-r","requirements.txt")
    $r1 = Invoke-Proc -File $py.Source -Args $pipArgs -TimeoutSec $PIP_TIMEOUT -StepName "pip_install"
    $steps += $r1
    Append-Log ("[pip_install] exit={0}`nSTDOUT:`n{1}`nSTDERR:`n{2}" -f $r1.ExitCode,$r1.StdOut,$r1.StdErr)
    if ($r1.TimedOut) {
      $digest = "TIMEOUT_" + (Make-Digest16 "pip_install" $r1.ExitCode $r1.StdOut $r1.StdErr)
      return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
    }
    if ($r1.ExitCode -ne 0) {
      $digest = (Make-Digest16 "pip_install" $r1.ExitCode $r1.StdOut $r1.StdErr)
      return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
    }
  }

  # Step 2: ruff lint (if available)
  $ruffCmd = (Get-Command ruff -ErrorAction SilentlyContinue)
  if ($ruffCmd) {
    try {
      if ($env:SKIP_LINT -eq "1") {
        Append-Log "Skip lint by SKIP_LINT=1"
      } else {
        # Limit files to avoid command line length issues
        $pyFiles = Get-ChildItem -Recurse -Include *.py -File | Where-Object { 
          $_.FullName -notmatch "\\\.(git|hg|svn|__pycache__|\.mypy_cache|\.pytest_cache|\.ruff_cache|\.venv|venv|env|build|dist|node_modules|data|datasets|logs|notebooks|analysis|archive|reports)\\" 
        } | Select-Object -First 50 | ForEach-Object { $_.FullName }
        
        if ($pyFiles.Count -eq 0) {
          Append-Log "No Python files found for lint."
        } else {
          $common = @("check","--quiet","--no-cache","--force-exclude")
          $cfg = Join-Path $ROOT "ruff.toml"
          if (Test-Path $cfg) {
            $args = $common + $pyFiles
          } else {
            $excludeList = ".git,.hg,.svn,__pycache__,.mypy_cache,.pytest_cache,.ruff_cache,.venv,venv,env,build,dist,node_modules,data,datasets,logs,notebooks,analysis,archive,reports"
            $args = $common + @(
              "--select","E,F,W,I",
              "--ignore","E501",
              "--line-length","120",
              "--target-version","py311",
              "--extend-exclude",$excludeList
            ) + $pyFiles
          }
          $r2 = Invoke-Proc -File $ruffCmd.Source -Args $args -TimeoutSec $LINT_TIMEOUT -StepName "lint"
          $steps += $r2
          Append-Log ("[lint] exit={0}`nSTDOUT:`n{1}`nSTDERR:`n{2}" -f $r2.ExitCode,$r2.StdOut,$r2.StdErr)
          if ($r2.TimedOut) {
            $lintFail = $true
            $digest = "TIMEOUT_" + (Make-Digest16 "lint" $r2.ExitCode $r2.StdOut $r2.StdErr)
            return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
          }
          if ($r2.ExitCode -ne 0) {
            $lintFail = $true
            $digest = (Make-Digest16 "lint" $r2.ExitCode $r2.StdOut $r2.StdErr)
            return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
          }
        }
      }
    } catch {
      $lintFail = $true
      Append-Log ("Ruff exception: {0}" -f $_.Exception.Message)
      $digest = (Make-Digest16 "lint" -1 "" $_.Exception.Message)
      return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
    }
  } else {
    Append-Log "ruff not found; skip lint."
  }

  # Step 3: pytest (smoke by default; FULL_TEST=1 to run all)
  $pytestCmd = (Get-Command pytest -ErrorAction SilentlyContinue)
  if ($pytestCmd) {
    # Set Python module path to include project root
    $env:PYTHONPATH = $ROOT
    
    $RUN_SMOKE = $true
    if ($env:FULL_TEST -eq "1") { $RUN_SMOKE = $false }
    $args = @("-q","--maxfail=1","--disable-warnings","--color=no","--durations=10","--rootdir",$ROOT)
    if ($RUN_SMOKE) { $args = @("-q","-m","smoke","-k","smoke","--maxfail=1","--disable-warnings","--color=no","--durations=10","--rootdir",$ROOT) }

    # Run tests only if tests exist, excluding problematic directories
    $hasTests = (Test-Path (Join-Path $ROOT "tests")) -or ((Get-ChildItem -Path $ROOT -Recurse -Filter "test_*.py" -File | Where-Object { 
      $_.FullName -notmatch "\\\\(analysis\\\\(past|old)|past|old)\\\\" 
    } | Measure-Object).Count -gt 0)
    
    if ($hasTests) {
      # Add ignore patterns for problematic directories
      $ignoreArgs = @("--ignore=analysis/past/", "--ignore=analysis/old/", "--ignore=past/", "--ignore=old/")
      $args = $args + $ignoreArgs
      
      $r3 = Invoke-Proc -File $pytestCmd.Source -Args $args -TimeoutSec $TEST_TIMEOUT -StepName "pytest"
      $steps += $r3
      Append-Log ("[pytest] exit={0}`nSTDOUT:`n{1}`nSTDERR:`n{2}" -f $r3.ExitCode,$r3.StdOut,$r3.StdErr)
      if ($r3.TimedOut) {
        $testFail = $true
        $digest = "TIMEOUT_" + (Make-Digest16 "pytest" $r3.ExitCode $r3.StdOut $r3.StdErr)
        return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
      }
      if ($r3.ExitCode -ne 0) {
        $testFail = $true
        $digest = (Make-Digest16 "pytest" $r3.ExitCode $r3.StdOut $r3.StdErr)
        return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
      }
    } else {
      Append-Log "No tests detected; skip pytest."
    }
  } else {
    Append-Log "pytest not found; skip tests."
  }

  # Summary
  $pass = -not ($lintFail -or $testFail)
  if (-not $pass -and -not $digest -and $steps.Count -gt 0) {
    $last = $steps[-1]
    $digest = (Make-Digest16 $last.Step $last.ExitCode $last.StdOut $last.StdErr)
  }
  return @{ pass=$pass; lintFail=$lintFail; testFail=$testFail; digest=$digest; log=$LOG }
}

# -----------------------------
# Execute & Output
# -----------------------------
$result = Invoke-QuickCheck

if ($result.pass) {
  Write-Host "=== QUICK_CHECK_RESULT: PASS ==="
} else {
  Write-Host "=== QUICK_CHECK_RESULT: FAIL ==="
  $dg = if ($result.digest) { $result.digest } else { "UNKNOWN_" + (Get-Date -Format HHmmss) }
  Write-Host ("=== QUICK_CHECK_FAIL_DIGEST: {0} ===" -f $dg)
}
Write-Host ("LOG: {0}" -f $result.log)

# Machine-readable JSON
$payload = @{
  pass     = $result.pass
  lintFail = $result.lintFail
  testFail = $result.testFail
  digest   = $result.digest
  log      = $result.log
}
Write-Host "=== QUICK_CHECK_JSON_BEGIN ==="
$payload | ConvertTo-Json -Compress | Write-Host
Write-Host "=== QUICK_CHECK_JSON_END ==="
Write-Host "=== QUICK_CHECK_COMPLETE ==="

if ($result.pass) { exit 0 } else { exit 1 }
