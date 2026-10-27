param(
    [int]$Epochs = 1,
    [int]$ImageSize = 416,
    [string]$Device = "cpu",
    [string]$Batch = "16",
    [string]$Dataset = "Rubbish_data_set",
    [string]$Format = "yolo",
    [string[]]$Classes = @(),
    [string]$Model = "",
    [switch]$SkipTransform,
    [switch]$SkipValidate,
    [switch]$SkipIdentityCheck,
    [switch]$DryRun
)

# =============================================================================
# Rubbish 数据集训练脚本
#
# 用法（必须在仓库根目录，或 -File 用绝对路径）：
#   powershell -ExecutionPolicy Bypass -File .\scripts\train_rubbish.ps1 -Epochs 10 -Device 0
#
# 前置条件（脚本会逐项检查并给出提示）：
#   1. 一个装好 od-platform 的解释器：
#        D:\Anaconda\envs\credit_risk\python.exe -m pip install -e .\apps\platform
#   2. 原始数据集：data\raw\<Dataset>\（YOLO 格式为 images\ + labels\）
#   3. 权重：models\pretrained\yolov8n.pt 或 yolo11n.pt（缺省会自动挑一个）
# =============================================================================

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$yamlPath = "apps\platform\configs\datasets\$Dataset.yaml"
$rawDir = Join-Path $repoRoot "data\raw\$Dataset"

if (-not $Classes -or $Classes.Count -eq 0) {
    # Rubbish 数据集的类别名是 class_0..class_9；YOLO 格式必须显式给出顺序
    $Classes = 0..9 | ForEach-Object { "class_$_" }
}

# ---------------------------------------------------------------------------
# 找训练解释器：conda 的 ODP → PATH → conda 注册表 / 常见安装位置
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
        if ($trimmed) { $candidates += (Join-Path $trimmed "python.exe") }
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

$python = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $ok = $false
    try {
        & $candidate -c "import od_platform, ultralytics, torch" *> $null
        $ok = ($LASTEXITCODE -eq 0)
    } catch {
        $ok = $false
    }
    if ($ok) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    $installHint = "No Python interpreter with ODPlatform training dependencies was found.`n"
    $installHint += "请先在装有 ultralytics + torch 的环境里安装本项目的平台包：`n"
    $installHint += "  D:\Anaconda\envs\credit_risk\python.exe -m pip install -e .\apps\platform`n"
    $installHint += "（如果用 conda：conda activate ODP 后执行 python -m pip install -e `".\apps\platform[dev]`"）"
    throw $installHint
}

# ---------------------------------------------------------------------------
# 选权重：yolov8n.pt → yolo11n.pt → 让 ultralytics 自动下载
# ---------------------------------------------------------------------------
if (-not $Model) {
    $yolov8 = Join-Path $repoRoot "models\pretrained\yolov8n.pt"
    $yolo11 = Join-Path $repoRoot "models\pretrained\yolo11n.pt"
    if (Test-Path -LiteralPath $yolov8) {
        $Model = $yolov8
    } elseif (Test-Path -LiteralPath $yolo11) {
        $Model = $yolo11
    } else {
        $Model = "yolo11n"
    }
}

# ---------------------------------------------------------------------------
# 预检：数据集目录、图片是否有效（0 字节图片是最常见的坑）
# ---------------------------------------------------------------------------
$datasetProblem = $null
$emptyImages = @()
if (-not (Test-Path -LiteralPath $rawDir)) {
    $datasetProblem = "原始数据集目录不存在：$rawDir`n请把数据集放到 data\raw\$Dataset\（YOLO 格式：images\ + labels\）。"
} else {
    $images = @(Get-ChildItem -LiteralPath $rawDir -Recurse -File -Include *.jpg, *.jpeg, *.png, *.bmp -ErrorAction SilentlyContinue)
    if ($images.Count -eq 0) {
        $datasetProblem = "目录 $rawDir 里没有图片（jpg/png/jpeg/bmp）。"
    } else {
        $emptyImages = @($images | Where-Object { $_.Length -eq 0 })
        if ($emptyImages.Count -gt 0) {
            $datasetProblem = ("有 {0}/{1} 张图片是 0 字节（多为 Git LFS 未拉取或复制损坏），例如：`n  {2}`n无法用于训练，请重新获取数据。" -f `
                $emptyImages.Count, $images.Count, $emptyImages[0].FullName)
        }
    }
}

Write-Host "Using Python: $python"
Write-Host "Dataset:      $rawDir"
Write-Host "Dataset yaml: $yamlPath"
Write-Host "Model:        $Model"
Write-Host "Epochs:       $Epochs   ImageSize: $ImageSize   Batch: $Batch   Device: $Device"

if ($DryRun) {
    if ($datasetProblem) { Write-Host "Dataset check: FAILED" -ForegroundColor Yellow; Write-Host $datasetProblem -ForegroundColor Yellow }
    else { Write-Host "Dataset check: OK" -ForegroundColor Green }
    Write-Host "Dry run finished (nothing executed)."
    exit 0
}

if ($datasetProblem) {
    throw $datasetProblem
}

$env:PYTHONUTF8 = "1"
Push-Location $repoRoot
try {
    if (-not $SkipTransform) {
        Write-Host "Transforming and freezing dataset $Dataset ..."
        $transformArgs = @(
            "-m", "od_platform.cli.transform_data",
            "--dataset", $Dataset,
            "--format", $Format,
            "--task", "detect",
            "--train-rate", "0.8",
            "--val-rate", "0.1",
            "--seed", "42",
            "--split-strategy", "stratified_multilabel",
            "--classes"
        ) + $Classes
        & $python @transformArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset transformation failed."
        }
    }

    if (-not $SkipValidate) {
        Write-Host "Validating dataset ..."
        & $python -m od_platform.cli.validate_data $yamlPath
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset validation failed. Training stopped."
        }
    }

    Write-Host "Starting training: epochs=$Epochs imgsz=$ImageSize batch=$Batch device=$Device"
    $trainArgs = @(
        "-m", "od_platform.cli.train_model",
        "--config", "train",
        "--data", $Dataset,
        "--model", $Model,
        "--epochs", $Epochs,
        "--batch", $Batch,
        "--imgsz", $ImageSize,
        "--device", $Device,
        "--experiment-name", "rubbish_yolo"
    )
    if ($SkipIdentityCheck) {
        # 演示/小数据集常因 split 样本过少被判 INFO，训练入口默认拒绝；明确知情后跳过
        Write-Host "WARNING: -SkipIdentityCheck 已开启，将跳过数据身份核验（审计会标记）。" -ForegroundColor Yellow
        $trainArgs += "--skip-validate"
    }
    & $python @trainArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
