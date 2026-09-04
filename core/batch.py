# -*- coding: utf-8 -*-
"""
批量导入导出：扫描文件夹内图片与视频，按目标语言分文件夹导出。
输出结构：输出目录/俄语/xxx.jpg、输出目录/西语/xxx.mp4 ...
"""
import os
import shutil
import tempfile

from .translate_engine import TranslateEngine, SUPPORTED_TARGETS
from .image_translate import translate_image
from .video_translate import translate_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def scan_media(input_dir):
    """扫描文件夹，返回 (图片列表, 视频列表)，保持相对路径。"""
    images, videos = [], []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(root, f)
            rel = os.path.relpath(full, input_dir)
            if ext in IMAGE_EXTS:
                images.append((full, rel))
            elif ext in VIDEO_EXTS:
                videos.append((full, rel))
    images.sort(key=lambda x: x[1])
    videos.sort(key=lambda x: x[1])
    return images, videos


def run_batch(input_dir, output_dir, targets,
              do_images=True, do_videos=True, keep_background=True,
              cover_original=True, replace_text=True,
              log=print, progress=None, cancel_check=None):
    """
    批量翻译主入口。
    targets: ['ru','es','pt'] 子集
    progress: 回调 (done, total, message)
    cancel_check: 返回 True 表示用户取消
    """
    images, videos = scan_media(input_dir)
    log(f"[批量] 扫描完成：图片 {len(images)} 张、视频 {len(videos)} 个，目标语言 {len(targets)} 种")
    if not images and not videos:
        try:
            _all = []
            for _r, _d, _fs in os.walk(input_dir):
                _all.extend(_fs)
            log(f"[批量] 文件夹内实际文件（前 20 个）：{', '.join(_all[:20]) or '（空）'}")
        except Exception:
            pass
    tasks = []
    if do_images:
        tasks += [("image", x) for x in images]
    if do_videos:
        tasks += [("video", x) for x in videos]
    total = len(tasks) * len(targets)
    if total == 0:
        log("[批量] 未找到可处理的图片或视频")
        return {"done": 0, "failed": 0}

    engine = TranslateEngine(log=log)
    done, failed = 0, 0
    # Windows 下 OpenCV/ffmpeg 对中文路径支持不佳：非 ASCII 路径先复制到临时目录
    staging = tempfile.mkdtemp(prefix="translate_stage_")
    seq = [0]

    def _is_ascii(s):
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def _stage_in(path):
        if _is_ascii(path):
            return path
        seq[0] += 1
        tmp = os.path.join(staging, f"in_{seq[0]}{os.path.splitext(path)[1]}")
        shutil.copyfile(path, tmp)
        return tmp

    def _stage_out(path):
        if _is_ascii(path):
            return path, None
        seq[0] += 1
        tmp = os.path.join(staging, f"out_{seq[0]}{os.path.splitext(path)[1]}")
        return tmp, path
    for target in targets:
        lang_name = SUPPORTED_TARGETS[target]
        for kind, (full, rel) in tasks:
            if cancel_check and cancel_check():
                log("[批量] 用户取消")
                return {"done": done, "failed": failed}
            out_path = os.path.join(output_dir, lang_name, rel)
            try:
                real_out, final_out = _stage_out(out_path)
                if kind == "image":
                    translate_image(_stage_in(full), real_out, target, engine=engine, log=log)
                else:
                    translate_video(_stage_in(full), real_out, target, engine=engine,
                                    log=log, keep_background=keep_background,
                                    cover_original=cover_original)
                if final_out:
                    os.makedirs(os.path.dirname(final_out), exist_ok=True)
                    shutil.move(real_out, final_out)
                    # 视频还有同名 .srt 字幕文件
                    srt_tmp = os.path.splitext(real_out)[0] + ".srt"
                    if os.path.exists(srt_tmp):
                        shutil.move(srt_tmp, os.path.splitext(final_out)[0] + ".srt")
                done += 1
            except Exception as e:
                failed += 1
                log(f"[批量] 失败 {rel} ({lang_name}): {e}")
            if progress:
                progress(done + failed, total, rel)
    log(f"[批量] 完成：成功 {done}，失败 {failed}，共 {total}")
    return {"done": done, "failed": failed}
