param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $repoRoot "apps\web-backend"

# ---------------------------------------------------------------------------
# 找解释器：优先 conda 里的 ODP 环境 → 当前 PATH 上的 python → conda 注册表
# 与常见 Anaconda / Miniconda 安装位置（本机 conda 未加入 PATH 时也能启动）
# ---------------------------------------------------------------------------
$candidates = @()

if (Get-Command conda -ErrorAction SilentlyContinue) {
    $odpLine = (& conda env list 2>$null | Where-Object { $_ -match "^\s*ODP\s+" } | Select-Object -First 1)
    if ($odpLine -match "([A-Za-z]:\\.+?)\s*$") {
        $candidates += (Join-Path $Matches[1].Trim() "python.exe")
    }
}

$currentPython = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($currentPython) {
    $candidates += $currentPython
}

$envFile = Join-Path $HOME ".conda\environments.txt"
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in (Get-Content -LiteralPath $envFile -Encoding UTF8)) {
        $trimmed = $line.Trim()
        if (-not $trimmed) { continue }
        $candidates += (Join-Path $trimmed "python.exe")
    }
}

$roots = @(
    (Join-Path $HOME "anaconda3"),
    (Join-Path $HOME "miniconda3"),
    (Join-Path $HOME "miniforge3"),
    "C:\ProgramData\anaconda3",
    "C:\ProgramData\miniconda3",
    "D:\Anaconda",
    "D:\Miniconda3"
)
foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    $candidates += (Join-Path $root "python.exe")
    $envsDir = Join-Path $root "envs"
    if (Test-Path -LiteralPath $envsDir) {
        foreach ($envDir in (Get-ChildItem -LiteralPath $envsDir -Directory -ErrorAction SilentlyContinue)) {
            $candidates += (Join-Path $envDir.FullName "python.exe")
        }
    }
}

function Test-PythonImports {
    param([string]$Python, [string]$Code)
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    try {
        & $Python -c $Code *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$python = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-PythonImports -Python $candidate -Code "import django, rest_framework, corsheaders") {
        $python = $candidate
        break
    }
}

if (-not $python) {
    throw 'No Python interpreter with Django dependencies was found. Run: python -m pip install -r ".\apps\web-backend\requirements.txt"'
}

# ---------------------------------------------------------------------------
# 找 YOLO 解释器（可能与 Web 环境不同），通过 ODP_YOLO_PYTHON 传给后端
# ---------------------------------------------------------------------------
$yoloPython = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-PythonImports -Python $candidate -Code "import ultralytics, cv2") {
        $yoloPython = $candidate
        break
    }
}

Push-Location $webRoot
try {
    $env:PYTHONUTF8 = "1"
    Write-Host "Using web Python:  $python"
    if ($yoloPython) {
        $env:ODP_YOLO_PYTHON = $yoloPython
        Write-Host "Using YOLO Python: $yoloPython"
    } else {
        Write-Host "YOLO Python:       not found (install ultralytics + opencv-python, or set ODP_YOLO_PYTHON in .env)"
    }

    & $python manage.py migrate --noinput
    if ($LASTEXITCODE -ne 0) {
        throw "Database migration failed."
    }

    Write-Host "ODPlatform Web: http://127.0.0.1:$Port/"
    Write-Host "Django Admin:  http://127.0.0.1:$Port/admin/"
    & $python manage.py runserver "127.0.0.1:$Port" --noreload
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
