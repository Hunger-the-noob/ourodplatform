"""新增：逐张校验图片容器及像素解码，识别空文件、截断文件和损坏图片。"""

from __future__ import annotations

from PIL import Image

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.common.progress import track
from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check


@check("image_integrity")
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    issues = []
    items = [(split, image) for split in snap.splits for image in snap.images_per_split[split]]
    for split, image in track(items, "image_integrity"):
        try:
            with Image.open(image) as decoded:
                decoded.verify()
            # verify 只验证容器，重新打开并 load 才能验证像素是否可解码。
            with Image.open(image) as decoded:
                decoded.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            issues.append(
                {"split": split, "file": image.name, "kind": "decode_error", "message": str(exc)}
            )
    return CheckResult(
        "image_integrity",
        CheckSeverity.ERROR if issues else CheckSeverity.PASS,
        f"{len(issues)}/{len(items)} 张图片无法完整解码"
        if issues
        else f"全部 {len(items)} 张图片可完整解码",
        {
            "total_images": len(items),
            "n_bad": len(issues),
            "bad_images_preview": issues[:DETAILS_PREVIEW_LIMIT],
        },
        issues=issues,
    )
