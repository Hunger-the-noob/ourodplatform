# 阅读副本：新增第 16 项质检，先 verify 再 load，捕获空文件、伪造图片和截断图片。
"""图片须同时通过容器校验和像素解码；任意坏图阻止训练。"""

import warnings

from PIL import Image

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.common.progress import iter_progress
from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check

_NAME = "image_integrity"


@check(_NAME)
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    bad_images = []
    n_bad = n_total = 0
    for split, images in ctx.snapshot.images_per_split.items():
        for path in iter_progress(images, f"D4 decode {split} images"):
            n_total += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        image.verify()
                    # verify 不读取所有像素；重开并 load 才能发现截断的 JPEG 等。
                    with Image.open(path) as image:
                        image.load()
            except (
                OSError,
                ValueError,
                SyntaxError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as error:
                n_bad += 1
                if len(bad_images) < DETAILS_PREVIEW_LIMIT:
                    bad_images.append(
                        {
                            "split": split,
                            "image": str(path),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    details = {"total_images": n_total, "n_bad": n_bad, "bad_images_preview": bad_images}
    if n_bad:
        return CheckResult(
            _NAME, CheckSeverity.ERROR, f"{n_bad}/{n_total} 张图片损坏或无法解码", details
        )
    return CheckResult(_NAME, CheckSeverity.PASS, f"全部 {n_total} 张图片可解码", details)
