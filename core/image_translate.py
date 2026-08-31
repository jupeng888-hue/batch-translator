# -*- coding: utf-8 -*-
"""
图片翻译管线：OCR 识别 -> 翻译 -> 智能抹除原文 -> 按原排版回填译文
"""
import os
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .translate_engine import TranslateEngine

_ocr_instance = None


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_instance = RapidOCR()
    return _ocr_instance


# ---------- 字体 ----------

_FONT_CANDIDATES = [
    # Windows 常见（支持西里尔/拉丁）
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    # Linux（测试环境）
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # 项目自带字体（打包时放置）
    os.path.join(os.path.dirname(__file__), "..", "assets", "NotoSans-Regular.ttf"),
]


def _find_font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_FONT_PATH = _find_font_path()


_FONT_BOLD_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    os.path.join(os.path.dirname(__file__), "..", "assets", "NotoSans-Bold.ttf"),
]

_FONT_BOLD_PATH = next((p for p in _FONT_BOLD_CANDIDATES if os.path.exists(p)), None)


# ---------- 多语种渲染（字体 + 复杂文字整形） ----------
_RENDER_LANG = "en"


def set_render_language(lang):
    global _RENDER_LANG
    _RENDER_LANG = lang


# 各文字系统字体候选（Windows 优先，其次 Linux/项目自带）
_FONT_SCRIPT_TABLE = {
    "ja": (["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/YuGothR.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
           ["C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/YuGothB.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]),
    "ko": (["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
           ["C:/Windows/Fonts/malgunbd.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]),
    "ar": (["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
           ["C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    "th": (["C:/Windows/Fonts/leelawui.ttf", "C:/Windows/Fonts/angsana.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
           ["C:/Windows/Fonts/leelawdb.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
}


def _script_font_path(lang, bold):
    entry = _FONT_SCRIPT_TABLE.get(lang)
    if not entry:
        return None
    cands = entry[1] if bold else entry[0]
    p = next((c for c in cands if os.path.exists(c)), None)
    if p is None and bold:
        p = next((c for c in entry[0] if os.path.exists(c)), None)
    return p


def _load_font(size, bold=False):
    sp = _script_font_path(_RENDER_LANG, bold)
    if sp:
        try:
            if _RENDER_LANG in ("ar", "th"):
                return ImageFont.truetype(sp, size, layout_engine=ImageFont.Layout.RAQM)
            return ImageFont.truetype(sp, size)
        except Exception:
            pass
    if bold and _FONT_BOLD_PATH:
        return ImageFont.truetype(_FONT_BOLD_PATH, size)
    if _FONT_PATH:
        return ImageFont.truetype(_FONT_PATH, size)
    return ImageFont.load_default()


def _shape_for_render(text):
    """阿语：字母连写整形 + 从右到左重排。"""
    if _RENDER_LANG == "ar":
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


# ---------- OCR 结果处理 ----------

def _box_to_rect(box):
    """四点框 -> 外接矩形 (x1, y1, x2, y2)"""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


_CJK_RE = None


def contains_cjk(text):
    """文本是否含中日韩表意字符（只有含中文的框才需要翻译）。"""
    global _CJK_RE
    if _CJK_RE is None:
        import re
        _CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
    return bool(_CJK_RE.search(text))


def _expand_ocr_boxes(keep, img_w, img_h):
    """OCR 框常漏掉首尾字符边缘（粗体美术字尤其明显），向外扩一圈再抹除，
    避免原字残留形成黑痕；扩框时避开相邻文字框。"""
    rects = [_box_to_rect(b) for b, _ in keep]
    out = []
    for i, (b, t) in enumerate(keep):
        x1, y1, x2, y2 = rects[i]
        bw, bh = x2 - x1, y2 - y1
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        for frac in (1.0, 0.5, 0.0):
            ex, ey = bw * 0.35 * frac, bh * 0.45 * frac
            cx1, cy1 = max(0, x1 - ex), max(0, y1 - ey)
            cx2, cy2 = min(img_w, x2 + ex), min(img_h, y2 + ey)
            clash = any(
                j != i and not (cx2 <= rects[j][0] or cx1 >= rects[j][2]
                                or cy2 <= rects[j][1] or cy1 >= rects[j][3])
                for j in range(len(rects)))
            if not clash:
                nx1, ny1, nx2, ny2 = cx1, cy1, cx2, cy2
                break
        out.append(([(nx1, ny1), (nx2, ny1), (nx2, ny2), (nx1, ny2)], t))
    return out


def merge_line_boxes(items):
    """把同一行、间距很小的 OCR 框合并为一个语义块（提升翻译连贯性与排版）。
    仅合并高度相近的框，避免把标题与邻近小字错误合并。"""
    if not items:
        return items
    enriched = []
    for box, text in items:
        r = _box_to_rect(box)
        enriched.append({"box": box, "text": text, "rect": r,
                         "cy": (r[1] + r[3]) / 2, "h": max(8, r[3] - r[1])})
    enriched.sort(key=lambda e: (e["cy"], e["rect"][0]))
    merged = []
    for e in enriched:
        if merged:
            last = merged[-1]
            lr = last["rect"]
            min_h = min(last["h"], e["h"])
            max_h = max(last["h"], e["h"])
            v_close = abs(e["cy"] - (lr[1] + lr[3]) / 2) < min_h * 0.6
            gap = e["rect"][0] - lr[2]
            # 只允许向右顺序拼接（同一行内、间距小）；
            # 按 cy 排序时同行靠左的框会产生负间距，绝不合并（多栏清单会跨栏串行）
            h_close = -2 <= gap < min_h * 0.6
            similar_h = max_h <= min_h * 1.4
            if v_close and h_close and similar_h:
                x1 = min(lr[0], e["rect"][0]); y1 = min(lr[1], e["rect"][1])
                x2 = max(lr[2], e["rect"][2]); y2 = max(lr[3], e["rect"][3])
                last["rect"] = (x1, y1, x2, y2)
                last["text"] = last["text"] + " " + e["text"]
                last["box"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                last["h"] = max(8, y2 - y1)
                continue
        merged.append(dict(e))
    return [(m["box"], m["text"]) for m in merged]


def _estimate_colors(img_bgr, rect):
    """估计文字颜色：以外圈背景色为基准，框内与背景差异最大的像素簇为文字色。"""
    x1, y1, x2, y2 = rect
    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return (0, 0, 0), (255, 255, 255)
    inliers, bg = _ring_inliers(img_bgr, rect, pad=8)
    if bg is None:
        bg = np.median(region.reshape(-1, 3), axis=0)
    pixels = region.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(pixels - bg, axis=1)
    thresh = np.percentile(dist, 75)
    fg_px = pixels[dist >= thresh]
    if len(fg_px) == 0:
        return tuple(int(c) for c in bg), tuple(int(c) for c in bg)
    fg = np.median(fg_px, axis=0)
    # 文字覆盖率不足 25% 时（宽扁框/多栏合并框），p75 取到的全是背景，
    # 估计出的"文字色"≈背景色 → 改看差异最大的前 5% 像素簇
    if int(max(fg) - min(fg)) <= 30 and np.abs(fg - bg).max() <= 30:
        th0 = np.percentile(dist, 95)
        fg_px0 = pixels[dist >= th0]
        if len(fg_px0) >= 10:
            fg = np.median(fg_px0, axis=0)
    # 防混色：p75 估计落在中间灰，或亮度接近/高于背景（亮底上不可能有"白字"）时，
    # 多半是深色笔画+浅色描边/图标混在一起；改看差异最大的前 5% 像素簇，取真正的文字本色
    _lum = _luminance(fg)
    _blum = _luminance(bg)
    if 90 < _lum < 250 and _blum > _lum - 10:
        th2 = np.percentile(dist, 95)
        fg_px2 = pixels[dist >= th2]
        if len(fg_px2) >= 10:
            fg2 = np.median(fg_px2, axis=0)
            _lum2 = _luminance(fg2)
            if _lum2 < 70 or _lum2 > 225 or (_blum > 200 and _lum2 < _lum - 40):
                fg = fg2
    return tuple(int(c) for c in fg), tuple(int(c) for c in bg)  # BGR


# ---------- 抹字 ----------

def _ring_inliers(img_bgr, rect, pad=8):
    """矩形外圈像素的内点（剔除污染的文字笔画）。返回 (内点像素, 中位色)。"""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = rect
    ring = []
    for (rx1, ry1, rx2, ry2) in [
        (x1 - pad, y1 - pad, x2 + pad, y1),
        (x1 - pad, y2, x2 + pad, y2 + pad),
        (x1 - pad, y1, x1, y2),
        (x2, y1, x2 + pad, y2),
    ]:
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        if rx2 > rx1 and ry2 > ry1:
            ring.append(img_bgr[ry1:ry2, rx1:rx2].reshape(-1, 3))
    if not ring:
        return None, None
    px = np.concatenate(ring).astype(np.float32)
    med = np.median(px, axis=0)
    dist = np.abs(px - med).mean(axis=1)
    inliers = px[dist < 25]
    if len(inliers) < len(px) * 0.5:
        return None, None  # 背景复杂，内点占比不足
    return inliers, np.median(inliers, axis=0)


def _gradient_fill(img, rect, pad=8, feather=15):
    """对整个【加边后的】矩形按行在左右外圈中位色间插值填充，保留渐变；
    边缘羽化过渡，消除可见的矩形边界。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
    inliers, med = _ring_inliers(img, rect, pad)
    if med is None:
        return False
    fill = img.copy()
    lx1, lx2 = max(0, x1 - pad), x1
    rx1, rx2 = x2, min(w, x2 + pad)
    for y in range(y1p, y2p):
        colors = []
        for (sx1, sx2) in ((lx1, lx2), (rx1, rx2)):
            if sx2 > sx1:
                px = img[y, sx1:sx2].astype(np.float32)
                d = np.abs(px - med).mean(axis=1)
                good = px[d < 25]
                colors.append(np.median(good, axis=0) if len(good) else med)
            else:
                colors.append(med)
        lc, rc = colors
        n = x2p - x1p
        t = np.linspace(0, 1, n)[:, None]
        fill[y, x1p:x2p] = (lc * (1 - t) + rc * t).astype(np.uint8)
    # 羽化合成
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1p:y2p, x1p:x2p] = 255
    k = max(9, feather | 1)
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    img[:] = (fill * alpha + img * (1 - alpha)).astype(np.uint8)
    return True


def _bg_is_smooth(img_bgr, rect, pad=8, thresh=18):
    """背景是否平滑：外圈内点像素相对中位色的离散程度。"""
    inliers, med = _ring_inliers(img_bgr, rect, pad)
    if inliers is None:
        return False
    return np.abs(inliers - med).mean(axis=1).std() < thresh


_LAMA = None
_LAMA_FAILED = False


def _get_lama(log=print):
    """懒加载 LaMa 修复模型（首次约 200MB 下载，仅一次）。"""
    global _LAMA, _LAMA_FAILED
    if _LAMA is not None or _LAMA_FAILED:
        return _LAMA
    try:
        from simple_lama_inpainting import SimpleLama
        log("[图片] 加载 LaMa 无痕修复模型（首次需下载约 200MB）…")
        _LAMA = SimpleLama()
    except Exception as e:
        log(f"[图片] LaMa 不可用，回退传统修复算法：{e}")
        _LAMA_FAILED = True
    return _LAMA


def _detect_label(img_bgr, rect):
    """检测框内的实心彩色标签（粉/绿/橙等圆角色块）。
    返回 ((lx1,ly1,lx2,ly2), 标签颜色) 或 None。"""
    x1, y1, x2, y2 = rect
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] > 60).astype(np.uint8)
    if sat_mask.mean() < 0.2:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sat_mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy, cw, ch, area = stats[i]
    rh, rw = sub.shape[:2]
    if cw < rw * 0.6 or ch < rh * 0.5 or area < rw * rh * 0.3:
        return None  # 标签必须主导整个文字框（排除双色横幅等）
    if area / max(1, cw * ch) < 0.5:
        return None  # 不是实心色块
    col = tuple(int(c) for c in np.median(sub[lab == i].reshape(-1, 3), axis=0))
    return (x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch), col


def _label_stroke_mask(img_bgr, rect, label_col, region_mask, inner=None):
    """彩色标签内只抹文字笔画：取与标签色差异大的像素。
    inner 为未膨胀文字框：质心落在框外的连通块（如标签上部的数字/小数点）保留不抹。"""
    x1, y1, x2, y2 = rect
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return region_mask
    d = np.abs(sub.astype(np.float32) - np.array(label_col, dtype=np.float32)).max(axis=2)
    binary = (d > 70).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    rh, rw = sub.shape[:2]
    keep = np.zeros_like(binary)
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 6:
            continue
        if area > rw * rh * 0.5:
            continue
        if inner is not None:
            _gy = y1 + cy + ch / 2
            if _gy < inner[1]:
                continue  # 文字行上方的内容（如标签上部的数字）保留不抹
        # 圆点状小组件（小数点/句点）保留：笔画组件不会是这么饱满的近似正方形
        if area <= 150 and 0.5 <= cw / max(1, ch) <= 2.0 and area >= 0.5 * cw * ch:
            continue
        keep[lab == i] = 255
    if int(keep.sum()) < 200:
        return region_mask
    keep = cv2.dilate(keep, np.ones((7, 7), np.uint8))
    if inner is not None:
        # 裁剪到文字行上缘以下：膨胀晕不许侵入文字行上方（数字/小数点区域）
        _cut = max(0, inner[1] + 15 - y1)
        keep[:_cut, :] = 0
    out = np.zeros_like(region_mask)
    out[y1:y2, x1:x2] = keep
    return out


def _bg_uniformity(img_bgr, rect, bg_bgr):
    """框内背景均匀度：接近估计背景色的像素占比。"""
    x1, y1, x2, y2 = rect
    sub = img_bgr[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    if sub.size == 0:
        return 0.0
    d = np.abs(sub - np.array(bg_bgr, dtype=np.float32)).max(axis=1)
    return float((d <= 45).mean())


def _detect_border(img_bgr, rect):
    """检测框内的装饰边框（描边圆角框等），返回其外接矩形；无则 None。"""
    x1, y1, x2, y2 = rect
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    fg, _ = _estimate_colors(img_bgr, rect)
    fl = _luminance(fg)
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # 主用 Otsu 自适应阈值（按文字明暗选极性），能完整覆盖渐变/半透明笔画；
    # 退化（占比过小/过大）时回退固定边距阈值
    _flag = cv2.THRESH_BINARY_INV if fl < 128 else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, _flag + cv2.THRESH_OTSU)
    _ratio = float((binary > 0).mean())
    if _ratio < 0.01 or _ratio > 0.6:
        if fl < 128:
            binary = (gray <= min(255, int(fl) + 95)).astype(np.uint8) * 255
        else:
            binary = (gray >= max(0, int(fl) - 95)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    rh, rw = sub.shape[:2]
    bx1 = by1 = 10 ** 9
    bx2 = by2 = -1
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        fill_ratio = area / max(1, cw * ch)
        if cw >= rw * 0.6 and ch >= rh * 0.6 and fill_ratio < 0.4 and area >= 60:
            bx1 = min(bx1, cx); by1 = min(by1, cy)
            bx2 = max(bx2, cx + cw); by2 = max(by2, cy + ch)
    if bx2 < 0:
        return None
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 30 or bh < 12 or bw > rw * 1.05:
        return None
    return (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)


def _stroke_mask(img_bgr, rect, region_mask, k=7, fg=None, inner=None):
    """在框内按文字颜色提取笔画级 mask：只抹文字笔画，保住装饰边框/底色。
    提取失败（如白字彩底标签）时回退整框 mask。
    fg 可传入按未膨胀框估计的文字色（膨胀区含图标/邻行时重估极易混色误判极性）。
    inner 为未膨胀框：连通块质心落在 inner 之外（如框上方的图标）一律保留不抹。"""
    x1, y1, x2, y2 = rect
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return region_mask
    if fg is None:
        fg, _ = _estimate_colors(img_bgr, rect)
    fl = _luminance(fg)
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # 主用 Otsu 自适应阈值（按文字明暗选极性），能完整覆盖渐变/半透明笔画；
    # 退化（占比过小/过大）时回退固定边距阈值
    _flag = cv2.THRESH_BINARY_INV if fl < 128 else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, _flag + cv2.THRESH_OTSU)
    _ratio = float((binary > 0).mean())
    if _ratio < 0.01 or _ratio > 0.6:
        if fl < 128:
            binary = (gray <= min(255, int(fl) + 95)).astype(np.uint8) * 255
        else:
            binary = (gray >= max(0, int(fl) - 95)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    rh, rw = sub.shape[:2]
    keep = np.zeros_like(binary)
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 8:
            continue
        fill_ratio = area / max(1, cw * ch)
        if cw >= rw * 0.9 and ch >= rh * 0.85 and fill_ratio < 0.30:
            continue  # 贴边的细线框 = 装饰边框，保留不抹（文字行虽高但填充率高，不受影响）
        if area > rw * rh * 0.6:
            continue  # 超大块 = 底色区域，保留
        if inner is not None:
            # 质心落在未膨胀框之外 = 框外内容（上方图标/邻行文字），保留不抹
            _gx, _gy = x1 + cx + cw / 2, y1 + cy + ch / 2
            if not (inner[0] <= _gx <= inner[2] and inner[1] <= _gy <= inner[3]):
                continue
        keep[labels == i] = 255
    if int((keep > 0).sum()) < 2:
        return region_mask  # 提取不到笔画，回退整框
    keep = cv2.dilate(keep, np.ones((k, k), np.uint8))
    out = np.zeros_like(region_mask)
    out[y1:y2, x1:x2] = keep
    return out


def erase_text_lama(img_bgr, boxes, dilate=10, log=print):
    """LaMa AI 无痕抹字；失败返回 None 由调用方回退。"""
    lama = _get_lama(log)
    if lama is None:
        return None
    try:
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        _unis = []
        _flat_fills = []
        for box in boxes:
            pts = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            x1 = max(0, int(pts[:, 0].min()) - dilate)
            y1 = max(0, int(pts[:, 1].min()) - dilate)
            x2 = min(w, int(pts[:, 0].max()) + dilate)
            y2 = min(h, int(pts[:, 1].max()) + dilate)
            ux1 = max(0, int(pts[:, 0].min()))
            uy1 = max(0, int(pts[:, 1].min()))
            ux2 = min(w, int(pts[:, 0].max()))
            uy2 = min(h, int(pts[:, 1].max()))
            _fg_e, _bg_e = _estimate_colors(img_bgr, (ux1, uy1, ux2, uy2))
            fl, bl = _luminance(_fg_e), _luminance(_bg_e)
            uni = _bg_uniformity(img_bgr, (ux1, uy1, ux2, uy2), _bg_e)
            sat = int(max(_bg_e)) - int(min(_bg_e))
            stroke_k = None
            label = _detect_label(img_bgr, (ux1, uy1, ux2, uy2))
            if label is not None:
                # 彩色标签：只抹标签内的文字笔画，标签底色完整保留
                (lx1, ly1, lx2, ly2), lcol = label
                box_mask = np.zeros_like(mask)
                box_mask[ly1:ly2, lx1:lx2] = 255
                mask |= _label_stroke_mask(img_bgr, (lx1, ly1, lx2, ly2), lcol, box_mask,
                                           inner=(ux1, uy1, ux2, uy2))
                continue
            if uni >= 0.5:
                if fl < 110 and bl > 140:
                    stroke_k = 13   # 深字浅底：笔画抹除，膨胀加大盖住半透明残边
                elif fl > 170:
                    stroke_k = 15   # 白字：连阴影/描边一起抹
            elif fl > 170 and bl < 140:
                # 白字压深色照片背景：整块抹除会留污斑，改用笔画抹除
                stroke_k = 15
            if stroke_k is not None:
                _unis.append(uni)
                box_mask = np.zeros_like(mask)
                box_mask[y1:y2, x1:x2] = 255
                _sm = _stroke_mask(img_bgr, (x1, y1, x2, y2), box_mask, k=stroke_k, fg=_fg_e,
                                   inner=(ux1, uy1, ux2, uy2))
                mask |= _sm
                # 高均匀浅底：LaMa 容易留雾状残影，记下mask稍后按背景色平涂
                if uni >= 0.65 and _luminance(_bg_e) > 200:
                    _flat_fills.append((_sm, _bg_e))
            else:
                _unis.append(uni)
                mask[y1:y2, x1:x2] = 255
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = cv2.cvtColor(np.array(lama(Image.fromarray(rgb), Image.fromarray(mask))), cv2.COLOR_RGB2BGR)
        out = out[:h, :w]  # SimpleLama 内部会把尺寸对齐到 8 的倍数，裁剪回原尺寸
        # 二次抹除：照片/渐变背景上 LaMa 可能留下淡淡的笔画回声，
        # 在修复结果上重新检测一遍残余笔画，有则再抹一次
        try:
            mask2 = np.zeros_like(mask)
            for box in boxes:
                pts = np.asarray(box, dtype=np.float64).reshape(-1, 2)
                x1 = max(0, int(pts[:, 0].min()) - dilate)
                y1 = max(0, int(pts[:, 1].min()) - dilate)
                x2 = min(w, int(pts[:, 0].max()) + dilate)
                y2 = min(h, int(pts[:, 1].max()) + dilate)
                # 彩色标签/装饰边框区域跳过：二抹会误伤标签底色
                if _detect_label(img_bgr, (max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min())),
                                         min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max())))) is not None:
                    continue
                _fg2, _bg2 = _estimate_colors(img_bgr, (x1, y1, x2, y2))
                _fl2, _bl2 = _luminance(_fg2), _luminance(_bg2)
                if not ((_fl2 < 110 and _bl2 > 140) or _fl2 > 170):
                    continue
                # 残影检测：直接与背景色比亮度（深字找比背景暗的，白字找比背景亮的），
                # 比在修复结果上重做 Otsu 笔画提取稳定得多（近均匀浅底上 Otsu 阈值漂移大）
                _reg2 = np.zeros_like(mask)
                _gray2 = cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
                # 浅底（>240）残影与底色仅差十几级，阈值收紧到 12 才能抓到
                _m2 = 12 if _bl2 > 240 else 25
                if _fl2 < 110:
                    _echo = (_gray2 < int(_bl2) - _m2).astype(np.uint8) * 255
                    _k2 = 13
                else:
                    _echo = (_gray2 > int(_bl2) + _m2).astype(np.uint8) * 255
                    _k2 = 15
                # 滤掉触及区域外缘的连通块（图标/邻行文字等不属于本框残影的内容）
                _n2, _lab2, _st2, _ = cv2.connectedComponentsWithStats(_echo, 8)
                _eh, _ew = _echo.shape[:2]
                _keep2 = np.zeros_like(_echo)
                for _i in range(1, _n2):
                    _cx, _cy, _cw, _ch, _ca = _st2[_i]
                    if _ca < 6:
                        continue
                    if _cx <= 1 or _cy <= 1 or _cx + _cw >= _ew - 1 or _cy + _ch >= _eh - 1:
                        continue  # 贴边 = 框外内容侵入，保留不抹
                    _keep2[_lab2 == _i] = 255
                _echo = cv2.dilate(_keep2, np.ones((_k2, _k2), np.uint8))
                _reg2[y1:y2, x1:x2] = _echo
                mask2 |= _reg2
            if int((mask2 > 0).sum()) > 300:
                rgb2 = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                out = cv2.cvtColor(np.array(lama(Image.fromarray(rgb2), Image.fromarray(mask2))), cv2.COLOR_RGB2BGR)
                out = out[:h, :w]
                mask = cv2.bitwise_or(mask, mask2)
                log("[图片] 检测到笔画残影，已二次抹除")
        except Exception:
            pass
        # 高均匀浅底平涂：LaMa 在这类背景上的填充色总会差几级，直接涂背景色无痕
        for _fm, _fb in _flat_fills:
            out[_fm > 0] = _fb
        # 边缘羽化融合：修复区与原图平滑过渡，消除矩形色块接缝
        m = mask.astype(np.float32) / 255.0
        m = cv2.dilate(m, np.ones((13, 13), np.uint8))  # 保证文字区完全不透原图
        m = cv2.GaussianBlur(m, (0, 0), 8)
        # 修复区整体柔化仅在照片类复杂背景下启用（模拟焦外虚化遮 LaMa 色带）；
        # 浅色均匀背景上启用反而把邻近图标/线条糊成雾斑
        if _unis and min(_unis) < 0.5:
            out_soft = cv2.GaussianBlur(out, (0, 0), 5)
        else:
            out_soft = out
        res = out_soft.astype(np.float32) * m[..., None] + img_bgr.astype(np.float32) * (1.0 - m[..., None])
        res = res.astype(np.uint8)
        # 高均匀浅底二次定色：羽化环会把原图文字边缘按比例混回形成雾状残影，
        # 均匀底色上直接把笔画区（含外晕）整体定成背景色，彻底杜绝残影
        for _fm, _fb in _flat_fills:
            _core = cv2.dilate((_fm > 0).astype(np.uint8), np.ones((25, 25), np.uint8))
            res[_core > 0] = _fb
        return res
    except Exception as e:
        log(f"[图片] LaMa 修复失败，回退传统算法：{e}")
        return None


def erase_text(img_bgr, boxes, blend=True, smooth_thresh=18, dilate_factor=0.18, light=False):
    """逐框智能抹字：平滑/渐变背景做逐行插值填充，复杂背景修复填充+羽化模糊融合。
    blend=False 时保留修复锐度（适合视频小字条，避免模糊压平背景结构）。
    返回 (抹字后图像, 每框是否平滑背景的标记列表)。"""
    out = img_bgr.copy()
    smooth_flags = []
    mask_inpaint = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    inpaint_radius = 5
    blur_kernel = 5
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        rect = _box_to_rect(box)
        box_h = max(8, rect[3] - rect[1])
        pad = max(6, int(box_h * 0.15))
        smooth = _bg_is_smooth(img_bgr, rect, pad=pad, thresh=smooth_thresh)
        if smooth:
            tmp = out.copy()
            if _gradient_fill(tmp, rect, pad=pad):
                out = tmp
                smooth_flags.append(True)
                continue
        smooth_flags.append(False)
        # 复杂背景：NS 修复
        m = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [pts], 255)
        k = max(7, int(box_h * dilate_factor))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)
        mask_inpaint = cv2.bitwise_or(mask_inpaint, m)
        inpaint_radius = max(inpaint_radius, min(15, box_h // 4))
        blur_kernel = max(blur_kernel, int(box_h * 0.5) | 1)
    if mask_inpaint.any():
        out = cv2.inpaint(out, mask_inpaint, inpaint_radius, cv2.INPAINT_NS)
        if blend:
            # 羽化模糊融合：修复区域与周围虚化的照片背景自然衔接，遮盖修复涂抹痕迹
            blur_kernel = min(blur_kernel, 51)
            feather_k = max(21, (min(out.shape[0], out.shape[1]) // 25) | 1)
            if light:
                # 轻量模式：只柔化修复边缘，不压平背景（视频小字条用）
                blur_kernel = max(9, min(blur_kernel, 15))
                feather_k = 15
            blurred = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), 0)
            feather = cv2.GaussianBlur(mask_inpaint, (feather_k, feather_k), 0).astype(np.float32) / 255.0
            feather = feather[:, :, None]
            out = (blurred * feather + out * (1 - feather)).astype(np.uint8)
    return out, smooth_flags


# ---------- 排版回填 ----------

def _wrap_text_to_width(draw, text, font, max_width):
    """按像素宽度断行（西文按词优先，超长词硬断）。"""
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            # 超长单词硬断
            while word and draw.textlength(word, font=font) > max_width:
                cut = len(word)
                while cut > 1 and draw.textlength(word[:cut], font=font) > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _wrap_balanced(draw, text, font, max_width):
    """平衡断行：同样行数下让各行宽度尽量均匀，避免最后一行只挂一两个词。"""
    base = _wrap_text_to_width(draw, text, font, max_width)
    words = text.split(" ")
    k = len(base)
    if k <= 1 or len(words) < 3:
        return base
    import functools
    n = len(words)
    widths = [draw.textlength(w, font=font) for w in words]
    space_w = draw.textlength(" ", font=font)

    @functools.lru_cache(None)
    def lw(i, j):
        return sum(widths[i:j + 1]) + space_w * (j - i)

    avg = lw(0, n - 1) / k
    INF = float("inf")
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
    bp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for j in range(1, n + 1):
        for l in range(1, k + 1):
            for i in range(l - 1, j):
                if dp[i][l - 1] == INF:
                    continue
                w = lw(i, j - 1)
                if w > max_width:
                    continue
                # 孤词行重罚（如 "filter" 单独挂一行）
                orphan = (avg * 1.5) ** 2 if (j - i) == 1 else 0
                cost = dp[i][l - 1] + (w - avg) ** 2 + orphan
                if cost < dp[j][l]:
                    dp[j][l] = cost
                    bp[j][l] = i
    if dp[n][k] == INF:
        return base
    lines = []
    j, l = n, k
    while j > 0:
        i = bp[j][l]
        lines.append(" ".join(words[i:j]))
        j, l = i, l - 1
    return lines[::-1]


def _estimate_bold(img_bgr, rect, fg_bgr):
    """估计原文字是否粗体：文字笔画最大厚度占框高比例 >10% 视为粗体。"""
    x1, y1, x2, y2 = rect
    region = img_bgr[max(0, y1):y2, max(0, x1):x2].astype(np.float32)
    if region.size == 0:
        return False
    fg = np.array(fg_bgr, dtype=np.float32)
    dist = np.linalg.norm(region - fg, axis=2)
    mask = (dist < 60).astype(np.uint8)
    if mask.sum() < 10:
        return False
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    stroke = float(dt.max()) * 2
    h = max(8, y2 - y1)
    return stroke / h > 0.10


def _sample_stroke(img_bgr, rect, fg_bgr):
    """采样原图描边色：取与文字色亮度相反的极端像素簇；
    若采样到的其实是背景色（说明原文无描边），返回 None 走中性描边。"""
    x1, y1, x2, y2 = rect
    region = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if region.size == 0:
        return None
    fg_lum = _luminance(fg_bgr)
    px = region.reshape(-1, 3).astype(np.float32)
    lums = 0.114 * px[:, 0] + 0.587 * px[:, 1] + 0.299 * px[:, 2]
    if fg_lum > 128:
        sel = px[lums <= np.percentile(lums, 18)]
    else:
        sel = px[lums >= np.percentile(lums, 82)]
    if len(sel) < 5:
        return None
    cand = np.median(sel, axis=0)
    if abs(_luminance(cand) - fg_lum) < 60:
        return None
    _in, bg = _ring_inliers(img_bgr, rect, pad=8)
    if bg is not None and abs(_luminance(cand) - _luminance(bg)) < 45:
        return None
    return tuple(int(c) for c in cand)


def _luminance(bgr):
    return 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]


def draw_text_block(img_pil, rect, text, fg_bgr, expand=1.35, margin=8,
                    left_limit=None, right_limit=None, tight=False, orig_bgr=None,
                    top_limit=None, bottom_limit=None, allow_widen=True,
                    label_mode=False, force_size=None, probe=False):
    """在矩形区域内回填译文：允许水平适度扩展，优先单行、贴近原字号。
    自动加描边（浅色文字配深描边、深色文字配浅描边），提升复杂背景可读性。"""
    x1, y1, x2, y2 = rect
    W, H = img_pil.size
    orig_w = max(4, x2 - x1)
    box_h = max(4, y2 - y1)
    cx = (x1 + x2) // 2
    # 以中心为基准扩大可用宽度（不超出图片边界、不侵入邻框）
    half = int(orig_w * expand) // 2
    ax1 = cx - half
    ax2 = cx + half
    ax1 = max(margin, ax1, left_limit if left_limit is not None else margin)
    ax2 = min(W - margin, ax2, right_limit if right_limit is not None else W - margin)
    if ax2 - ax1 < 8:  # 扩展被夹死则退回原框
        ax1, ax2 = x1, x2
    box_w = ax2 - ax1

    text = _shape_for_render(text)
    # 标签模式：文字必须完整落在标签内，高度余量收紧（多行也不行出标签）
    h_slack = 1.06 if label_mode else 1.3
    draw = ImageDraw.Draw(img_pil)
    fg_rgb = (fg_bgr[2], fg_bgr[1], fg_bgr[0])
    bold = _estimate_bold(orig_bgr, rect, fg_bgr) if orig_bgr is not None else False

    # 以原始框高为基准搜索合适字号与断行；字号设下限，避免译文缩得过小
    best = None
    single = None  # 单行候选：(font, lines, line_h, total_h, size)
    start = max(8, int(box_h * 0.95))
    floor = max(10, int(box_h * 0.30))
    if force_size is not None:
        # 同一水平带的文字统一字号（由调用方探测本组最小可行字号后强制指定）
        start = floor = max(8, int(force_size))
    n_words0 = len(text.split())
    cand = []  # 可行候选（字号从大到小）
    for size in range(start, floor - 1, -1):
        font = _load_font(size, bold=bold)
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * 1.12)
        if single is None and draw.textlength(text, font=font) <= box_w:
            single = (font, [text], line_h, line_h, size)
        # 断词保护：最长单词都放不下的字号直接跳过，宁可更小也不把单词截断
        if max((draw.textlength(_wd, font=font) for _wd in text.split()), default=0) > box_w:
            if best is None:
                best = (font, _wrap_balanced(draw, text, font, box_w), line_h, line_h * 9)
            continue
        lines = _wrap_balanced(draw, text, font, box_w)
        total_h = line_h * len(lines)
        max_line_w = max((draw.textlength(l, font=font) for l in lines), default=0)
        if max_line_w <= box_w and total_h <= box_h * h_slack:
            cand.append((font, lines, line_h, total_h))
            if not tight or len(cand) >= 6:
                break
        best = (font, lines, line_h, total_h)  # tight 模式：宁可小字也不溢出
    if single is None:
        # 常规字号都放不下单行时，继续向小字号找单行兜底（宁可字小也不溢出框）
        for size in range(floor - 1, 8, -1):
            f1 = _load_font(size, bold=bold)
            if draw.textlength(text, font=f1) <= box_w:
                a1, d1 = f1.getmetrics()
                lh1 = int((a1 + d1) * 1.12)
                single = (f1, [text], lh1, lh1, size)
                break
    if cand:
        chosen = cand[0]
        # 若最大字号方案存在孤词行，改用行数更少、无孤词且字号损失≤30% 的方案
        if n_words0 >= 3:
            def _orphans(c):
                return sum(1 for l in c[1] if len(l.split()) == 1)
            if _orphans(chosen) > 0:
                for c in cand[1:]:
                    if len(c[1]) < len(chosen[1]) and _orphans(c) == 0 \
                            and c[0].size >= chosen[0].size * 0.7:
                        chosen = c
                        break
        best = chosen
    font, lines, line_h, total_h = best
    # 断词兜底：选定字号下最长单词仍超宽时，逐级缩到单词完整放下为止
    if max((draw.textlength(_wd, font=font) for _wd in text.split()), default=0) > box_w:
        for _s in range(font.size - 1, 7, -1):
            _f = _load_font(_s, bold=bold)
            if max((draw.textlength(_wd, font=_f) for _wd in text.split()), default=0) <= box_w:
                font = _f
                lines = _wrap_balanced(draw, text, font, box_w)
                _a2, _d2 = font.getmetrics()
                line_h = int((_a2 + _d2) * 1.12)
                total_h = line_h * len(lines)
                break
    n_words = len(text.split())

    def _orphan_count(ls):
        if n_words < 3:
            return 0
        return sum(1 for l in ls if len(l.split()) == 1)

    # 短句（≤5 词）放不下单行、或当前断行存在孤词行时：
    # 适度加宽可用宽度重排（不超过邻框限制），优先单行，其次无孤词的更少行方案
    if allow_widen and len(lines) > 1 and n_words <= 6 and expand >= 1.0 \
            and (single is None or _orphan_count(lines) > 0 or n_words <= 3):
        lo = left_limit if left_limit is not None else margin
        hi = right_limit if right_limit is not None else W - margin
        cx = (ax1 + ax2) / 2
        improved = None
        for extra in (1.25, 1.5):
            half = box_w * extra / 2
            bw2 = min(cx + half, hi) - max(cx - half, lo)
            if bw2 <= box_w + 2:
                continue
            for size in range(start, floor - 1, -1):
                f2 = _load_font(size, bold=bold)
                a2, d2 = f2.getmetrics()
                lh2 = int((a2 + d2) * 1.12)
                if draw.textlength(text, font=f2) <= bw2:
                    improved = (f2, [text], lh2, lh2)
                    break
                ls2 = _wrap_balanced(draw, text, f2, bw2)
                mw2 = max((draw.textlength(l, font=f2) for l in ls2), default=0)
                if mw2 <= bw2 and lh2 * len(ls2) <= box_h * h_slack \
                        and len(ls2) < len(lines) and _orphan_count(ls2) == 0:
                    improved = (f2, ls2, lh2, lh2 * len(ls2))
                    break
            if improved is not None:
                break
        if improved is not None and improved[0].size >= font.size * 0.55:
            font, lines, line_h, total_h = improved
            single = None  # 已采用更优方案，不再走单行替换
    # 短句单行优先：一行能放下时尽量不换行；≤4 词的短句放宽字号损失容忍度
    # 标签模式（彩色小标签）除外：译文长、标签小，强制单行会缩到看不清，
    # 允许两行换取显著更大的字号
    if single is not None and len(lines) > 1 and not label_mode:
        s_font, s_lines, s_lh, s_th, s_size = single
        # ≤3 词短句：只要能挤进一行就不断行（字号可降到下限）
        thresh = 0.30 if n_words <= 3 else (0.40 if n_words <= 4 else 0.65)
        # 宽扁标签框（宽>2.2倍高）：多行必然压出框外，小字单行更干净
        wide_label = box_w >= box_h * 2.2 and s_size >= 12
        if s_size >= font.size * thresh or (wide_label and len(lines) > 1):
            font, lines, line_h, total_h = s_font, s_lines, s_lh, s_th
    if label_mode and single is not None and len(lines) > 1:
        # 单行字号若能达到多行方案的 75%，才用单行；否则保留更大的多行字
        s_font, s_lines, s_lh, s_th, s_size = single
        if s_size >= font.size * 0.75:
            font, lines, line_h, total_h = s_font, s_lines, s_lh, s_th

    # 最终保险：单行译文宽度不超出原框，避免压到装饰边框/框外区域
    if len(lines) == 1:
        lw1 = draw.textlength(lines[0], font=font)
        if lw1 > orig_w:
            for size in range(font.size - 1, 8, -1):
                f3 = _load_font(size, bold=bold)
                if draw.textlength(lines[0], font=f3) <= orig_w:
                    a3, d3 = f3.getmetrics()
                    line_h = int((a3 + d3) * 1.12)
                    font, total_h = f3, line_h
                    break
    if probe:
        return font.size  # 仅探测字号，不落笔
    # 描边颜色：优先继承原图描边色，否则用与文字亮度反差的中性色
    fg_lum = _luminance(fg_bgr)
    stroke_rgb = (30, 30, 30) if fg_lum > 128 else (240, 240, 240)
    if orig_bgr is not None:
        sampled = _sample_stroke(orig_bgr, rect, fg_bgr)
        if sampled is not None:
            stroke_rgb = (sampled[2], sampled[1], sampled[0])
    stroke_w = max(1, font.size // 18)

    # 垂直居中，水平居中；有纵向邻框时在可用空间内夹取，避免叠字
    ty = y1 + (box_h - total_h) // 2
    if top_limit is not None:
        ty = max(ty, top_limit)
    if bottom_limit is not None and ty + total_h > bottom_limit:
        ty = max(top_limit if top_limit is not None else 0, bottom_limit - total_h)
    for line in lines:
        lw = draw.textlength(line, font=font)
        tx = cx - lw // 2
        tx = max(margin, min(tx, W - margin - int(lw)))
        draw.text((tx, ty), line, font=font, fill=fg_rgb,
                  stroke_width=stroke_w, stroke_fill=stroke_rgb)
        ty += line_h
    return img_pil


# ---------- 主流程 ----------

def _imwrite_unicode(path, img):
    """cv2.imwrite 在 Windows 下不支持中文路径，改用 imencode + tofile。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(path)


def translate_image(image_path, output_path, target, engine=None, log=print):
    """翻译单张图片。target: 'ru'/'es'/'pt'"""
    engine = engine or TranslateEngine(log=log)
    # cv2.imread 在 Windows 下不支持中文路径，改用 imdecode
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    ocr = get_ocr()
    result, _ = ocr(image_path)
    if not result:
        log(f"[图片] 未检测到文字，原样复制: {os.path.basename(image_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _imwrite_unicode(output_path, img_bgr)
        return {"path": output_path, "texts": 0}

    keep = [(item[0], item[1].strip()) for item in result if item[1].strip()]
    # 只处理含中文的文字框；英文/数字/装饰字符原样保留
    keep = [(b, t) for b, t in keep if contains_cjk(t)]
    if not keep:
        log(f"[图片] 未检测到中文文字，原样复制: {os.path.basename(image_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _imwrite_unicode(output_path, img_bgr)
        return {"path": output_path, "texts": 0}
    keep = merge_line_boxes(keep)
    H_img, W_img = img_bgr.shape[:2]
    keep = _expand_ocr_boxes(keep, W_img, H_img)

    set_render_language(target)
    if hasattr(engine, "translate_with_image"):
        translations = engine.translate_with_image([t for _, t in keep], target, image_path)
    else:
        translations = engine.translate([t for _, t in keep], target)

    # 抹字：LaMa AI 无痕修复优先，失败回退传统算法（返回每框背景平滑标记）
    _erased = erase_text_lama(img_bgr, [b for b, _ in keep], log=log)
    if _erased is not None:
        erased, smooth_flags = _erased, [False] * len(keep)
    else:
        erased, smooth_flags = erase_text(img_bgr, [b for b, _ in keep])
    img_pil = Image.fromarray(cv2.cvtColor(erased, cv2.COLOR_BGR2RGB))

    # 回填
    orig_bgr = img_bgr
    rects = [_box_to_rect(b) for b, _ in keep]
    # 第一遍：确定每框的落字参数并探测各自可行字号；
    # 同一水平带（图标行/顶栏等）统一用组内最小字号，避免同排文字大小不一
    _jobs = []
    for (box, src_text), dst_text, rect, smooth in zip(keep, translations, rects, smooth_flags):
        fg, _bg = _estimate_colors(orig_bgr, rect)
        # 计算左右扩展限制：同一水平带上最近的邻框边缘
        x1, y1, x2, y2 = rect
        cy = (y1 + y2) / 2
        band = max(10, (y2 - y1) * 0.6)
        left_limit, right_limit = None, None
        top_limit, bottom_limit = None, None
        for other in rects:
            if other is rect:
                continue
            ox1, oy1, ox2, oy2 = other
            ocy = (oy1 + oy2) / 2
            if abs(ocy - cy) > band:
                continue  # 不在同一水平带
            if ox2 <= x1:  # 邻框在左侧
                left_limit = max(left_limit or 0, ox2 + 6)
            elif ox1 >= x2:  # 邻框在右侧
                right_limit = min(right_limit or 10 ** 9, ox1 - 6)
            # 纵向邻框（水平范围有交叠）：限制垂直落字空间
            if ox1 < x2 and ox2 > x1:
                if oy2 <= y1:
                    top_limit = max(top_limit or 0, oy2 + 4)
                elif oy1 >= y2:
                    bottom_limit = min(bottom_limit or 10 ** 9, oy1 - 4)
        # 平滑背景（彩色标签/徽标）：文字必须留在原框内，不外扩
        # 复杂背景（照片区域）：允许适度外扩
        expand = 1.05 if smooth else 1.35
        # 检测到装饰边框（描边圆角框等）：译文锁定在边框内部，不压框不越界
        allow_widen = True
        _fg_b, _bg_b = _estimate_colors(orig_bgr, rect)
        dark_on_light = _luminance(_fg_b) < 110 and _luminance(_bg_b) > 140 \
            and _bg_uniformity(orig_bgr, rect, _bg_b) >= 0.50
        border = _detect_border(orig_bgr, rect) if dark_on_light else None
        _label = _detect_label(orig_bgr, rect) if border is None else None
        if _label is not None:
            (lx1, ly1, lx2, ly2), _lcol = _label
            inx = max(2, int((lx2 - lx1) * 0.05))
            iny = max(1, int((ly2 - ly1) * 0.06))
            rect = (lx1 + inx, ly1 + iny, lx2 - inx, ly2 - iny)
            x1, y1, x2, y2 = rect
            expand = 1.0
            allow_widen = False
            # 文字色从标签内部重估：取与标签色差异最大的像素簇（即原文字色）
            _sub = orig_bgr[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
            if _sub.size:
                _d = np.abs(_sub - np.array(_lcol, dtype=np.float32)).max(axis=1)
                _sel = _sub[_d > 70]
                if len(_sel) >= 10:
                    fg = tuple(int(c) for c in np.median(_sel, axis=0))
        if border is not None:
            bx1, by1, bx2, by2 = border
            bw_b, bh_b = bx2 - bx1, by2 - by1
            if bw_b <= (x2 - x1) * 1.35 and bh_b <= (y2 - y1) * 1.6:
                inset_x = max(3, int(bw_b * 0.06))
                inset_y = max(2, int(bh_b * 0.10))
                rect = (bx1 + inset_x, by1 + inset_y, bx2 - inset_x, by2 - inset_y)
                x1, y1, x2, y2 = rect
                expand = 1.0
                allow_widen = False
        _size = draw_text_block(img_pil, rect, dst_text, fg, expand=expand,
                                left_limit=left_limit, right_limit=right_limit,
                                orig_bgr=orig_bgr, tight=True,
                                top_limit=top_limit, bottom_limit=bottom_limit,
                                allow_widen=allow_widen,
                                label_mode=_label is not None, probe=True)
        _jobs.append(dict(rect=rect, dst=dst_text, fg=fg, expand=expand,
                          left_limit=left_limit, right_limit=right_limit,
                          top_limit=top_limit, bottom_limit=bottom_limit,
                          allow_widen=allow_widen, label_mode=_label is not None,
                          size=_size if isinstance(_size, int) else None,
                          cy=(rect[1] + rect[3]) / 2, bh=rect[3] - rect[1]))

    # 同排统一字号：按纵向中心分组（带高相近），组内取最小可行字号
    _forced = [None] * len(_jobs)
    _used = [False] * len(_jobs)
    for i, a in enumerate(_jobs):
        if _used[i] or a["size"] is None:
            continue
        grp = [i]
        _used[i] = True
        for j in range(i + 1, len(_jobs)):
            b = _jobs[j]
            if _used[j] or b["size"] is None:
                continue
            if abs(b["cy"] - a["cy"]) <= max(a["bh"], b["bh"]) * 0.6 \
                    and min(a["bh"], b["bh"]) >= max(a["bh"], b["bh"]) * 0.65:
                grp.append(j)
                _used[j] = True
        if len(grp) > 1:
            _smin = min(_jobs[g]["size"] for g in grp)
            for g in grp:
                _forced[g] = _smin

    for j, jb in enumerate(_jobs):
        img_pil = draw_text_block(img_pil, jb["rect"], jb["dst"], jb["fg"],
                                  expand=jb["expand"],
                                  left_limit=jb["left_limit"], right_limit=jb["right_limit"],
                                  orig_bgr=orig_bgr, tight=True,
                                  top_limit=jb["top_limit"], bottom_limit=jb["bottom_limit"],
                                  allow_widen=jb["allow_widen"],
                                  label_mode=jb["label_mode"],
                                  force_size=_forced[j])

    out_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _imwrite_unicode(output_path, out_bgr)
    log(f"[图片] {os.path.basename(image_path)}: {len(keep)} 处文字已翻译 -> {target}")
    return {"path": output_path, "texts": len(keep)}
