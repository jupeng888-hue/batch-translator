# -*- coding: utf-8 -*-
"""ffmpeg 路径解析：系统 PATH -> imageio-ffmpeg 内置 -> static-ffmpeg 下载。"""
import shutil

_ffmpeg = None
_ffprobe = None


def _ensure(log=print):
    global _ffmpeg, _ffprobe
    if _ffmpeg:
        return _ffmpeg, (_ffprobe or _ffmpeg)
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if not (ff and fp):
        try:
            import static_ffmpeg
            log("[环境] 正在准备内置 ffmpeg…")
            static_ffmpeg.add_paths()
            ff2, fp2 = shutil.which("ffmpeg"), shutil.which("ffprobe")
            ff = ff or ff2
            fp = fp or fp2
        except Exception:
            pass
    if not ff:
        raise RuntimeError(
            "未找到 ffmpeg。请安装 ffmpeg 并加入 PATH，"
            "或保持联网让程序自动下载内置 ffmpeg。")
    _ffmpeg, _ffprobe = ff, (fp or ff)
    return _ffmpeg, _ffprobe


def ffmpeg_cmd(log=print):
    return _ensure(log)[0]


def ffprobe_cmd(log=print):
    return _ensure(log)[1]
