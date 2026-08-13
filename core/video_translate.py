# -*- coding: utf-8 -*-
"""
视频翻译管线：提取音频 -> ASR 时间轴 -> 翻译 -> 烧录字幕 -> AI 配音替换人声
配音保留背景音乐（人声/伴奏分离），字幕与配音时间轴对齐。
"""
import os
import re
import subprocess
import cv2
import tempfile
import asyncio
import edge_tts

from .translate_engine import TranslateEngine
from .ffmpeg_utils import ffmpeg_cmd, ffprobe_cmd

# edge-tts 音色（按效果优选的自然女声）
# 选用更富表现力的新一代音色（语速放慢后更自然有感情）
TTS_VOICES = {
    "en": "en-US-AvaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "es": "es-ES-ElviraNeural",
    "pt": "pt-BR-FranciscaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ar": "ar-EG-SalmaNeural",
    "th": "th-TH-PremwadeeNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "id": "id-ID-GadisNeural",
    "it": "it-IT-ElsaNeural",
    "tr": "tr-TR-EmelNeural",
}
BASE_RATE = "-10%"   # 基础语速放慢，接近真人讲解节奏
MAX_SPEEDUP = 15     # 译文偏长时最多加快 15%，避免机关枪式念白
MAX_ATEMPO = 1.15    # 末端微调上限，保证听感自然

_whisper_model = None


def get_whisper(log=print, model_size="small"):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        from .model_download import ensure_whisper_model
        path = ensure_whisper_model(model_size, log=log)
        _whisper_model = WhisperModel(path, device="cpu", compute_type="int8")
    return _whisper_model


# ---------- ffmpeg 辅助 ----------

def _run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd[:3])}...\n{p.stderr[-800:]}")
    return p


def extract_audio(video_path, wav_path, sample_rate=16000):
    _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", video_path,
          "-vn", "-ac", "1", "-ar", str(sample_rate), wav_path])


def get_duration(path):
    """用 ffmpeg -i 的 stderr 解析时长，兼容无 ffprobe 的环境。"""
    import re as _re
    p = _run([ffmpeg_cmd(), "-i", path], check=False)
    m = _re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", p.stderr)
    if not m:
        raise RuntimeError(f"无法获取时长: {path}")
    h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + s


# ---------- ASR ----------

def transcribe(video_path, workdir, log=print):
    """faster-whisper 识别，返回 [{start, end, text}]"""
    wav = os.path.join(workdir, "audio_16k.wav")
    extract_audio(video_path, wav)
    model = get_whisper(log=log)
    segments, info = model.transcribe(wav, language="zh", vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=500))
    out = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            out.append({"start": seg.start, "end": seg.end, "text": text})
    log(f"[视频] ASR 完成：{len(out)} 段语音")
    return out


# ---------- 字幕 ----------

def _srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n"
                    f"{seg['translation']}\n\n")


def burn_subtitles(video_path, srt_path, out_path, segments=None, log=print,
                   cover_original=True):
    """烧录字幕。cover_original=True 时在语音时段用半透明色块遮盖底部原硬字幕区域，
    译文显示在遮盖带上，避免与原字幕叠影。"""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    font_size = max(14, int(h * 0.040))

    filters = []
    if cover_original and segments:
        # 窄遮盖带：只盖住原硬字幕区域（典型位置 66%~84% 高度），不挡画面主体
        band_y = int(h * 0.66)
        band_h = int(h * 0.18)
        for seg in segments:
            filters.append(
                f"drawbox=x=0:y={band_y}:w={w}:h={band_h}:color=black@0.6:t=fill"
                f":enable='between(t,{seg['start']:.2f},{seg['end']:.2f})'")
    # 译文字幕贴底部，逐行深色底框，任何背景都清晰
    margin_v = int(h * 0.025)
    style = (f"FontName=DejaVu Sans,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
             f"OutlineColour=&H00000000,BackColour=&H90000000,BorderStyle=3,"
             f"Outline=1,Shadow=0,Alignment=2,MarginV={margin_v}")
    srt_esc = srt_path.replace("\\", "/").replace(":", "\\:")
    filters.append(f"subtitles='{srt_esc}':force_style='{style}'")
    vf = ",".join(filters)
    _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", video_path, "-vf", vf,
          "-c:v", "libx264", "-preset", "fast", "-crf", "20",
          "-c:a", "copy", "-pix_fmt", "yuv420p", out_path])
    log(f"[视频] 字幕已烧录" + ("（已遮盖原硬字幕）" if cover_original and segments else ""))


# ---------- 配音 ----------

async def _tts_one(text, voice, rate, out_path):
    tts = edge_tts.Communicate(text, voice, rate=rate)
    await tts.save(out_path)


def _tts_sync(text, voice, rate, out_path):
    asyncio.run(_tts_one(text, voice, rate, out_path))


def _audio_duration(path):
    return get_duration(path)


def _tts_retry(text, voice, rate, path, tries=3):
    import time as _time
    for t in range(tries):
        try:
            _tts_sync(text, voice, rate, path)
            return
        except Exception:
            if t == tries - 1:
                raise
            _time.sleep(2)


# 自适应语速候选档（从慢到快，优先慢速保证自然听感）
_RATE_CANDIDATES = ["-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+20%", "+25%"]
MAX_ATEMPO = 1.22  # 末端微调上限


def synthesize_dub_track(segments, target, total_duration, workdir, log=print):
    """为每段译文生成配音。

    自然听感优先的语速策略：
    - 严格窗口：每段必须在下一段开始前 120ms 结束，杜绝重叠抢话；
    - 从 -10% 慢速开始，逐档试探，取能放进窗口的最慢一档；
    - 裁掉 TTS 首尾空边，让语音贴齐画面节奏；
    - 极端短句才允许 atempo ≤1.22 的轻度微调。
    """
    import edge_tts  # noqa
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    voice = TTS_VOICES[target]
    track = AudioSegment.silent(duration=int(total_duration * 1000) + 1500)
    for i, seg in enumerate(segments):
        start_ms = seg["start"] * 1000
        slot_ms = (seg["end"] - seg["start"]) * 1000
        if i + 1 < len(segments):
            window = segments[i + 1]["start"] * 1000 - 120 - start_ms
        else:
            window = slot_ms * 1.3
        window = max(window, slot_ms * 0.8)
        mp3 = os.path.join(workdir, f"tts_{i}.mp3")
        _tts_retry(seg["translation"], voice, "-10%", mp3)
        d0 = _audio_duration(mp3) * 1000
        chosen = mp3
        if d0 > window:
            need = d0 / max(window, 1)
            est = -10 + int((need - 1) * 100) + 4
            vals = [int(r.rstrip('%')) for r in _RATE_CANDIDATES]
            idx = min(range(len(vals)), key=lambda k: abs(vals[k] - est))
            for k in range(idx, len(_RATE_CANDIDATES)):
                p2 = os.path.join(workdir, f"tts_{i}_r{k}.mp3")
                _tts_retry(seg["translation"], voice, _RATE_CANDIDATES[k], p2)
                d = _audio_duration(p2) * 1000
                chosen = p2
                if d <= window * 1.02:
                    break
        clip = AudioSegment.from_file(chosen)
        # 裁首尾静音
        rng = detect_nonsilent(clip, min_silence_len=70, silence_thresh=clip.dBFS - 22)
        if rng:
            s0 = max(0, rng[0][0] - 40)
            e0 = min(len(clip), rng[-1][1] + 40)
            clip = clip[s0:e0]
        if len(clip) > window * 1.05:
            ratio = len(clip) / window
            tmp = os.path.join(workdir, f"tts_{i}_fit.wav")
            clip.export(tmp, format="wav")
            fit = os.path.join(workdir, f"tts_{i}_fit2.wav")
            _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", tmp,
                  "-filter:a", f"atempo={min(ratio, MAX_ATEMPO):.3f}", fit])
            clip = AudioSegment.from_file(fit)
        track = track.overlay(clip, position=int(start_ms))
    dub_path = os.path.join(workdir, f"dub_{target}.wav")
    track.export(dub_path, format="wav")
    log(f"[视频] 配音轨生成完成（{len(segments)} 段）")
    return dub_path


def _save_wav(path, tensor, samplerate):
    """用标准库保存 (C, T) float tensor 为 16bit WAV，避免额外音频依赖。"""
    import wave
    import numpy as np
    arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    channels, frames = pcm.shape[0], pcm.shape[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.T.tobytes())


def separate_vocals(video_path, workdir, log=print):
    """人声/伴奏分离，返回伴奏路径。分离失败则返回 None（静音背景兜底）。"""
    try:
        from demucs.separate import Separator
        wav_full = os.path.join(workdir, "audio_full.wav")
        _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", video_path, "-vn",
              "-ar", "44100", "-ac", "2", wav_full])
        sep = Separator(model="htdemucs", device="cpu")
        _origin, separated = sep.separate_audio_file(wav_full)
        accomp = None
        for name, tensor in separated.items():
            if name == "vocals":
                continue
            accomp = tensor if accomp is None else accomp + tensor
        bg_path = os.path.join(workdir, "accompaniment.wav")
        _save_wav(bg_path, accomp, sep.samplerate)
        # 响度自校准：让伴奏接近原视频整体响度，避免分离后音乐过轻
        try:
            import re as _re
            def _rms_db(fp):
                q = _run([ffmpeg_cmd(), "-i", fp, "-af", "astats=metadata=0", "-f", "null", "-"],
                         check=False)
                m = _re.search(r"RMS level dB:\s*(-?[\d.]+)", q.stderr)
                return float(m.group(1)) if m else None
            o_db, b_db = _rms_db(video_path), _rms_db(bg_path)
            if o_db is not None and b_db is not None:
                gain_db = max(-6.0, min(15.0, o_db - b_db - 1.0))
                if abs(gain_db) > 0.5:
                    boosted = os.path.join(workdir, "accompaniment_norm.wav")
                    _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", bg_path,
                          "-filter:a", f"volume={gain_db:.1f}dB", boosted])
                    bg_path = boosted
        except Exception:
            pass
        log("[视频] 人声分离完成，保留背景音")
        return bg_path
    except Exception as e:
        log(f"[视频] 人声分离不可用（{e}），配音将替换整条音轨")
        return None


def mux_final(video_with_subs, dub_path, bg_path, out_path, total_duration, log=print):
    """把烧录字幕的视频 + 配音 (+ 背景音) 合成最终视频。"""
    if bg_path:
        _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", video_with_subs,
              "-i", dub_path, "-i", bg_path,
              "-filter_complex",
              "[1][2]amix=inputs=2:weights='1 0.6':normalize=0[a]",
              "-map", "0:v", "-map", "[a]",
              "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])
    else:
        _run([ffmpeg_cmd(), "-y", "-v", "error", "-i", video_with_subs,
              "-i", dub_path, "-map", "0:v", "-map", "1:a",
              "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])
    log(f"[视频] 合成完成: {os.path.basename(out_path)}")


# ---------- 主流程 ----------

def translate_video(video_path, output_path, target, engine=None, log=print,
                    keep_background=True, cover_original=True, replace_text=True):
    """翻译单个视频：画面文字替换 + 字幕 + 配音。target: 'ru'/'es'/'pt'/'en'

    replace_text=True 时逐帧 OCR 画面中的中文，抹除后原位回填译文
    （视频看起来像原本就是目标语言拍的），此时无需再遮盖/烧录字幕。
    失败则自动回退到 遮盖原字幕 + 烧录译文字幕 方案。"""
    engine = engine or TranslateEngine(log=log)
    with tempfile.TemporaryDirectory(prefix="vtrans_") as workdir:
        # 0. 画面文字替换（逐帧）
        src_video = video_path
        replaced = False
        if replace_text:
            try:
                log("[视频] 逐帧处理画面文字（OCR→抹除→回填译文）…")
                clean = os.path.join(workdir, "frame_replaced.mp4")
                from .frame_text import replace_frame_text
                replace_frame_text(video_path, clean, engine, target, log)
                src_video = clean
                replaced = True
            except Exception as e:
                log(f"[视频] 画面文字替换失败（{e}），回退到字幕遮盖方案")
        # 1. ASR
        segments = transcribe(video_path, workdir, log=log)
        if not segments:
            log(f"[视频] 未检测到语音: {os.path.basename(video_path)}")
            return {"path": None, "segments": 0}
        # 2. 翻译
        translations = engine.translate([s["text"] for s in segments], target)
        for s, t in zip(segments, translations):
            s["translation"] = t
        # 3. 字幕（画面已替换文字则无需再烧字幕）
        srt = os.path.join(workdir, "translated.srt")
        write_srt(segments, srt)
        if replaced:
            subbed = src_video
            log("[视频] 画面文字已原位替换，跳过字幕烧录")
        else:
            subbed = os.path.join(workdir, "subtitled.mp4")
            burn_subtitles(src_video, srt, subbed, segments=segments, log=log,
                           cover_original=cover_original)
        # 4. 配音
        total = get_duration(video_path)
        dub = synthesize_dub_track(segments, target, total, workdir, log=log)
        # 5. 背景音（从原始视频分离，避免二重损失）
        bg = separate_vocals(video_path, workdir, log=log) if keep_background else None
        # 6. 合成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mux_final(subbed, dub, bg, output_path, total, log=log)
        # 同时导出 SRT 方便用户微调
        srt_out = os.path.splitext(output_path)[0] + ".srt"
        write_srt(segments, srt_out)
    return {"path": output_path, "segments": len(segments)}
