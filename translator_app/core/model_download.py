# -*- coding: utf-8 -*-
"""
模型下载管理：国内网络优先走 ModelScope，其次 HuggingFace 镜像/直连。
首次使用自动下载，之后离线可用。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_WHISPER_MS = "pengzhendong/faster-whisper-{size}"
_WHISPER_HF = "Systran/faster-whisper-{size}"
_NLLB_MS = "facebook/nllb-200-distilled-600M"
_NLLB_HF = "facebook/nllb-200-distilled-600M"


def _from_modelscope(model_id, log=print):
    from modelscope.hub.snapshot_download import snapshot_download
    log(f"[模型] 从 ModelScope 下载: {model_id}")
    return snapshot_download(model_id)


def _from_hf(model_id, log=print):
    from huggingface_hub import snapshot_download
    log(f"[模型] 从 HuggingFace 下载: {model_id}")
    return snapshot_download(model_id)


def _download(model_ms, model_hf, log=print):
    errors = []
    for name, fn in (("ModelScope", lambda: _from_modelscope(model_ms, log)),
                     ("HuggingFace", lambda: _from_hf(model_hf, log))):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            log(f"[模型] {name} 下载失败，尝试下一源")
    raise RuntimeError("模型下载失败：\n" + "\n".join(errors))


def ensure_whisper_model(size="small", log=print):
    """返回 faster-whisper 模型本地目录。"""
    return _download(_WHISPER_MS.format(size=size), _WHISPER_HF.format(size=size), log)


def ensure_nllb_model(log=print):
    """返回 NLLB-200 本地翻译模型目录。"""
    return _download(_NLLB_MS, _NLLB_HF, log)
