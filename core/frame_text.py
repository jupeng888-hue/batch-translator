"""逐帧处理视频画面文字：OCR 检测中文 -> 抹除 -> 回填译文。

采样帧做 OCR，帧间复用检测结果（文本通常在数帧内保持静止），
块位置做指数平滑避免抖动；OCR 结果跨语言缓存，多语言处理时只 OCR 一次。
"""
import os
import sys
import difflib
import subprocess

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ffmpeg_utils import ffmpeg_cmd
from core.image_translate import (contains_cjk, merge_line_boxes, erase_text,
                                  draw_text_block, _estimate_colors,
                                  set_render_language)

_SAMPLE_EVERY = 3          # 每隔几帧做一次 OCR
_RENDER_WINDOW = 6         # 一次检测后连续渲染的帧数（容忍单次漏检，避免原文闪回）
_EXPIRE = 12               # 超过这么多帧未再匹配则删除块
_IOU_MIN = 0.25
_TEXT_CHANGE_RATIO = 0.85   # 低于此相似度才考虑重译（配合连续确认抗抖动）


def _contrast_fg(img_bgr, rect):
    """根据抹除后区域的亮度选中性前景色：亮底深字、暗底白字。"""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return (30, 30, 30)
    med = np.median(region.reshape(-1, 3), axis=0)
    lum = 0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]
    return (30, 30, 30) if lum > 140 else (245, 245, 245)

_ocr_cache = {}            # (video_path, frame_idx) -> [(box4x2 list, text)]
_ocr_instance = [None]


def _get_ocr():
    if _ocr_instance[0] is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_instance[0] = RapidOCR()
    return _ocr_instance[0]


def _rect_int(rect):
    return tuple(int(round(v)) for v in rect)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


class _Block:
    __slots__ = ("rect", "text", "trans", "fg", "last_seen", "pending", "pending_n")

    def __init__(self, rect, text, trans, fg, seen):
        self.rect = rect          # [x0,y0,x1,y1] float, EMA 平滑
        self.text = text
        self.trans = trans
        self.fg = fg              # None 时渲染前按抹除背景亮度自适应黑/白
        self.last_seen = seen
        self.pending = None       # 疑似新文本，需连续两次相近识别才提交（抗 OCR 抖动）
        self.pending_n = 0


class FrameTextReplacer:
    def __init__(self, video_path, engine, target, log):
        self.video_path = video_path
        self.engine = engine
        self.target = target
        self.log = log
        self.blocks = []
        self.trans_cache = {}

    def _translate(self, text):
        if text not in self.trans_cache:
            self.trans_cache[text] = self.engine.translate([text], self.target)[0]
        return self.trans_cache[text]

    def _detect(self, frame, idx):
        key = (self.video_path, idx)
        if key in _ocr_cache:
            return _ocr_cache[key]
        result, _ = _get_ocr()(frame)
        out = []
        if result:
            for box, text, conf in result:
                # 视频不做跨框合并：角标/贴纸都是独立短词，合并会导致译文过长溢出
                if text and conf >= 0.4 and contains_cjk(text.strip()):
                    out.append((np.array(box, dtype=np.float32), text.strip()))
        _ocr_cache[key] = out
        return out

    def _update_blocks(self, frame, idx):
        detections = self._detect(frame, idx)
        for box, text in detections:
            xs, ys = box[:, 0], box[:, 1]
            rect = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            best, best_iou = None, _IOU_MIN
            for b in self.blocks:
                v = _iou(b.rect, rect)
                if v > best_iou:
                    best, best_iou = b, v
            if best is None:
                self.blocks.append(_Block(rect, text, self._translate(text),
                                          None, idx))
            else:
                best.rect = [0.5 * o + 0.5 * n for o, n in zip(best.rect, rect)]
                best.last_seen = idx
                ratio = difflib.SequenceMatcher(None, best.text, text).ratio()
                if ratio < _TEXT_CHANGE_RATIO:
                    from core.glossary import lookup_glossary
                    # 新读法命中词库而当前文本未命中：立即采用（词库译文质量最高）
                    if lookup_glossary(text, self.target) and not lookup_glossary(best.text, self.target):
                        best.text = text
                        best.trans = self._translate(text)
                        best.fg = None
                        best.pending, best.pending_n = None, 0
                    elif best.pending is not None and \
                            difflib.SequenceMatcher(None, best.pending, text).ratio() >= 0.7:
                        best.pending_n += 1
                    else:
                        best.pending, best.pending_n = text, 1
                    if best.pending_n >= 2:
                        best.text = text
                        best.trans = self._translate(text)
                        best.fg = None
                        best.pending, best.pending_n = None, 0
                else:
                    best.pending, best.pending_n = None, 0
        self.blocks = [b for b in self.blocks if idx - b.last_seen <= _EXPIRE]

    def process_frame(self, frame, idx):
        if idx % _SAMPLE_EVERY == 0:
            self._update_blocks(frame, idx)
        active = [b for b in self.blocks if idx - b.last_seen < _RENDER_WINDOW]
        if not active:
            return frame
        H, W = frame.shape[:2]

        def to_box(b):
            x0, y0, x1, y1 = b.rect
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                            dtype=np.float32)

        small = [b for b in active if b.rect[3] - b.rect[1] < 45]
        large = [b for b in active if b.rect[3] - b.rect[1] >= 45]
        img = frame
        smooth_map = {}
        if small:
            img, fl = erase_text(img, [to_box(b) for b in small], blend=True, light=True, dilate_factor=0.32, smooth_thresh=30)
            for b, f in zip(small, fl):
                smooth_map[id(b)] = f
        if large:
            img, fl = erase_text(img, [to_box(b) for b in large], blend=True, smooth_thresh=30)
            for b, f in zip(large, fl):
                smooth_map[id(b)] = f

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for b in active:
            sm = smooth_map.get(id(b), False)
            if b.fg is None:
                b.fg = _contrast_fg(img, _rect_int(b.rect))
            expand = 1.05 if sm else 1.15
            draw_text_block(pil, _rect_int(b.rect), b.trans, b.fg, expand,
                            left_limit=2, right_limit=W - 2, tight=True)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def replace_frame_text(video_path, out_path, engine, target, log):
    """生成画面文字已替换为译文的视频（无音轨）。"""
    set_render_language(target)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc = subprocess.Popen(
        [ffmpeg_cmd(), "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-an", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    replacer = FrameTextReplacer(video_path, engine, target, log)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = replacer.process_frame(frame, idx)
            proc.stdin.write(out.tobytes())
            idx += 1
            if idx % 150 == 0:
                log(f"    画面文字处理 {idx}/{total} 帧…")
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    log(f"    画面文字处理完成，共 {idx} 帧，替换词条 {len(replacer.trans_cache)} 条")
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("帧合成失败")
    return out_path
