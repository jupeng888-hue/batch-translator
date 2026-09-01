# -*- coding: utf-8 -*-
"""
混合翻译引擎：云端免费接口优先，本地 NLLB-200 模型兜底。
目标语言：俄语 ru / 西语 es / 葡语 pt
"""
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
import requests

# 国内网络环境下 HuggingFace 直连不通，统一走镜像站
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# hf-mirror 不支持 xet 协议，强制走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SUPPORTED_TARGETS = {
    "en": "英语", "ru": "俄语", "es": "西语", "pt": "葡语",
    "fr": "法语", "de": "德语", "ja": "日语", "ko": "韩语",
    "ar": "阿语", "th": "泰语", "vi": "越南语", "id": "印尼语",
    "it": "意大利语", "tr": "土耳其语",
}
# 智谱/LLM 提示用的英文语言名
LANG_EN_NAME = {
    "en": "English", "ru": "Russian", "es": "Spanish", "pt": "Portuguese",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "it": "Italian", "tr": "Turkish",
}
# NLLB 语言代码映射
NLLB_LANG = {
    "zh": "zho_Hans",
    "en": "eng_Latn", "ru": "rus_Cyrl", "es": "spa_Latn", "pt": "por_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "ar": "arb_Arab", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "it": "ita_Latn", "tr": "tur_Latn",
}


_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".batch_translator_config.json")


def _read_config_key():
    try:
        import json as _json
        return _json.load(open(_CONFIG_PATH, encoding="utf-8")).get("zhipu_key", "").strip()
    except Exception:
        return ""


def save_config_key(key):
    import json as _json
    try:
        cfg = {}
        if os.path.exists(_CONFIG_PATH):
            cfg = _json.load(open(_CONFIG_PATH, encoding="utf-8"))
        cfg["zhipu_key"] = key.strip()
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
    except Exception:
        pass


def _google_one(text, target, source="zh-CN", retries=2, timeout=15):
    """Google 网页免费接口，单条翻译。"""
    url = "https://translate.googleapis.com/translate_a/single"
    params = [("client", "gtx"), ("sl", source), ("tl", target), ("dt", "t"), ("q", text)]
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return "".join(seg[0] for seg in data[0] if seg and seg[0])
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _google_batch(texts, target, source="zh-CN", workers=4):
    """Google 接口并发批量翻译。"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: _google_one(t, target, source), texts))


class _TranSmart:
    """腾讯交互翻译 TranSmart 开放接口（国内直连，支持批量）。"""

    URL = "https://transmart.qq.com/api/imt"

    def translate(self, texts, target, source="zh", retries=2, timeout=15):
        payload = {
            "header": {
                "fn": "auto_translation",
                "session": str(uuid.uuid4()),
                "client_key": f"browser-chrome-110.0.0-{uuid.uuid4()}-{int(time.time() * 1000)}",
            },
            "type": "plain",
            "model_category": "normal",
            "text_domain": "general",
            "source": {"lang": source, "text_list": texts},
            "target": {"lang": target, "text_list": []},
        }
        for attempt in range(retries + 1):
            try:
                r = requests.post(self.URL, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                if data["header"].get("ret_code") != "succ":
                    raise RuntimeError(f"TranSmart 返回错误: {data['header']}")
                return data["auto_translation"]
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")


class _LocalNLLB:
    """本地 NLLB-200 蒸馏模型兜底（首次使用自动从镜像站下载，约 2.5GB）。"""

    MODEL_ID = "facebook/nllb-200-distilled-600M"

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        with self._lock:
            if self._model is None:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                from .model_download import ensure_nllb_model
                path = ensure_nllb_model()
                self._tokenizer = AutoTokenizer.from_pretrained(path)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(path)

    def translate(self, texts, target, source="zh"):
        self._ensure_loaded()
        tok, model = self._tokenizer, self._model
        tok.src_lang = NLLB_LANG[source]
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        forced = tok.convert_tokens_to_ids(NLLB_LANG[target])
        gen = model.generate(**inputs, forced_bos_token_id=forced, max_new_tokens=512)
        return tok.batch_decode(gen, skip_special_tokens=True)


class _Zhipu:
    """智谱 GLM（bigmodel.cn）：GLM-4V-Flash 看图翻译 / GLM-4-Flash 文本翻译。
    免费额度，需用户自填 API Key。"""

    URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_key, log=print):
        self.key = api_key
        self.log = log

    def _chat(self, messages, model, timeout=60):
        r = requests.post(self.URL, json={"model": model, "messages": messages},
                          headers={"Authorization": f"Bearer {self.key}"},
                          timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_array(content, n):
        import json as _json, re as _re
        m = _re.search(r"\[.*\]", content, _re.S)
        if not m:
            raise ValueError("GLM 返回不含 JSON 数组")
        arr = _json.loads(m.group(0))
        if len(arr) != n:
            raise ValueError(f"GLM 返回数量不符: {len(arr)} != {n}")
        out = []
        for x in arr:
            if isinstance(x, dict):  # 模型偶尔返回 {"text":..,"translation":..} 对象
                x = (x.get("translation") or x.get("dst") or x.get("target")
                     or x.get("译文") or list(x.values())[-1])
            out.append(str(x).strip())
        return out

    @staticmethod
    def _has_cjk(text):
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    @staticmethod
    def _strip_cjk(text):
        import re as _r
        return _r.sub(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+", " ", text).strip()

    _SCRIPT_RANGES = {
        "ru": ('\u0400', '\u04ff'), "ar": ('\u0600', '\u06ff'),
        "th": ('\u0e00', '\u0e7f'), "hi": ('\u0900', '\u097f'),
        "el": ('\u0370', '\u03ff'), "he": ('\u0590', '\u05ff'),
        "ko": ('\uac00', '\ud7af'), "ja": ('\u3040', '\u30ff'),
    }

    def _lang_suspect(self, arr, target):
        """目标语种有独立文字体系时判错语种：整批一个该体系字符都没有，或任何一条
        含拉丁字母却一个目标文字体系字符都没有（如俄语译文写成英语）= 模型答错语种。"""
        rng = self._SCRIPT_RANGES.get(target)
        if not rng:
            return False
        joined = "".join(arr)
        if not any(rng[0] <= c <= rng[1] for c in joined):
            return True
        for t in arr:
            has_target = any(rng[0] <= c <= rng[1] for c in t)
            has_latin = any('a' <= c.lower() <= 'z' for c in t)
            if has_latin and not has_target:
                return True  # 该条是纯拉丁文（英语等），目标语种应有独立文字
        return False

    def _hygiene(self, out, target):
        """非中日韩目标语种的译文不允许残留中文字符。残留时先单条重译（直接剔除
        会把"С前面 есть…"剔成病句），重译仍残留才剔除。"""
        if target in ("zh", "ja", "ko"):
            return out
        import json as _json
        fixed = []
        for t in out:
            if not self._has_cjk(t):
                fixed.append(t)
                continue
            t2 = None
            for _ in range(3):
                try:
                    lang = LANG_EN_NAME[target]
                    m2 = [{"role": "system", "content":
                           f"你是电商文案翻译引擎。把给定的中文电商文案翻译成地道、简洁的 {lang}，"
                           "像是母语运营写的。只输出译文文本，不要任何其他文字。"},
                          {"role": "user", "content": t}]
                    r = self._chat(m2, "glm-4-flash", 30).strip().strip('"').strip("'")
                    if r and not self._has_cjk(r) and not self._lang_suspect([r], target):
                        t2 = r
                        break
                except Exception:
                    pass
            if t2 is None:
                t2 = self._strip_cjk(t) or t
                self.log(f"[翻译] 译文含中文残留，重译失败已剔除: {t!r} -> {t2!r}")
            else:
                self.log(f"[翻译] 译文含中文残留，已单条重译: {t!r} -> {t2!r}")
            fixed.append(t2)
        return fixed

    def translate(self, texts, target, image_path=None, timeout=90):
        """texts: 中文短语列表。image_path 提供时走 GLM-4V 看图理解语境。"""
        import json as _json, base64 as _b64
        lang = LANG_EN_NAME[target]
        sys_prompt = (
            f"你是电商文案翻译引擎。把给定的中文电商文案翻译成地道、简洁的 {lang}，"
            "像是母语运营写的，不要直译腔。规则："
            "1) 保留数字、型号、单位（如 304、12小时、30 oz 按目标语言习惯书写）；"
            "2) 逐条对应，不多不少；3) 只输出 JSON 字符串数组（每个元素是纯译文文本，不要输出对象），不要任何其他文字；4) 输入文字来自 OCR 识别，可能有错别字，请按合理含义翻译；"
            f"5) 每一条都必须全部使用{lang}，严禁把任何一条译成英语或其他语言（原文本身的品牌英文名除外）。")
        user_text = "待翻译中文列表：\n" + _json.dumps(texts, ensure_ascii=False)
        img_ext = None
        if image_path:
            img_ext = os.path.splitext(image_path)[1].lstrip(".").lower().replace("jpg", "jpeg")
        result = self._translate_batch_robust(texts, target, sys_prompt, user_text,
                                              image_path, img_ext, timeout)
        if result is not None:
            return result
        raise RuntimeError("GLM 批量/单条翻译均失败")

    def _translate_batch_robust(self, texts, target, sys_prompt, user_text,
                                image_path=None, img_ext=None, timeout=90):
        """先试批量（2次），数量不符则逐条翻译，保证命中率。"""
        import json as _json, base64 as _b64
        if image_path:
            with open(image_path, "rb") as f:
                b64 = _b64.b64encode(f.read()).decode()
            messages = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/{img_ext};base64,{b64}"}},
                            {"type": "text",
                             "text": "这些是图中识别出的文字，结合图片语境翻译。\n" + user_text}]}]
            for _ in range(3):
                try:
                    arr = self._parse_array(self._chat(messages, "glm-4v-flash", timeout), len(texts))
                    if target not in ("zh", "ja", "ko") and any(self._has_cjk(x) for x in arr):
                        raise ValueError("译文混入中文，重试")
                    if self._lang_suspect(arr, target):
                        raise ValueError("译文非目标语种，重试")
                    return self._hygiene(arr, target)
                except Exception as e:
                    self.log(f"[翻译] GLM-4V 批量重试：{e}")
            # 4V 批量不行则转纯文本批量
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text + "\n\n注意：输出数组必须有且只有 "
                     + str(len(texts)) + " 个元素，与输入一一对应。"}]
        for _ in range(3):
            try:
                arr = self._parse_array(self._chat(messages, "glm-4-flash", timeout), len(texts))
                if target not in ("zh", "ja", "ko") and any(self._has_cjk(x) for x in arr):
                    raise ValueError("译文混入中文，重试")
                if self._lang_suspect(arr, target):
                    raise ValueError("译文非目标语种，重试")
                return self._hygiene(arr, target)
            except Exception as e:
                self.log(f"[翻译] GLM 批量重试：{e}")
        # 逐条兜底
        out = []
        try:
            for t in texts:
                m2 = [{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": "待翻译中文列表：\n" + _json.dumps([t], ensure_ascii=False)}]
                arr = self._parse_array(self._chat(m2, "glm-4-flash", timeout), 1)
                if (target not in ("zh", "ja", "ko") and self._has_cjk(arr[0])) \
                        or self._lang_suspect(arr, target):
                    arr = self._parse_array(self._chat(m2, "glm-4-flash", timeout), 1)
                out.append(arr[0])
            return self._hygiene(out, target)
        except Exception as e:
            self.log(f"[翻译] GLM 逐条翻译失败：{e}")
            return None



class TranslateEngine:
    """统一入口：云端优先（Google -> 腾讯TranSmart），失败自动降级本地模型。"""

    def __init__(self, log=print, zhipu_key=None):
        self.log = log
        self._transmart = _TranSmart()
        self._local = None
        key = zhipu_key or os.environ.get("ZHIPU_API_KEY") or _read_config_key()
        self._zhipu = _Zhipu(key, log) if key else None

    def _translate_batch(self, texts, target, source="zh-CN"):
        # 0) 智谱 GLM（有 Key 时优先，质量最好）
        if self._zhipu is not None:
            try:
                return self._zhipu.translate(texts, target)
            except Exception as e:
                self.log(f"[翻译] 智谱 GLM 不可用，切换免费接口：{e}")
        # 1) Google
        try:
            return _google_batch(texts, target, source)
        except Exception as e:
            self.log(f"[翻译] Google 接口不可用，切换腾讯 TranSmart：{e}")
        # 2) 腾讯 TranSmart
        try:
            return self._transmart.translate(texts, target, "zh")
        except Exception as e:
            self.log(f"[翻译] TranSmart 接口不可用，切换本地模型：{e}")
        # 3) 本地 NLLB
        if self._local is None:
            self._local = _LocalNLLB()
        return self._local.translate(texts, target, "zh")

    def translate_with_image(self, texts, target, image_path):
        """图片翻译专用：有智谱 Key 时看图翻译，否则走普通链路。"""
        assert target in SUPPORTED_TARGETS
        from .glossary import lookup_glossary
        results = [None] * len(texts)
        pending = []
        for i, t in enumerate(texts):
            hit = lookup_glossary(t, target)
            if hit:
                results[i] = hit
            else:
                pending.append((i, _pre_fix(t)))
        if pending:
            if self._zhipu is not None:
                try:
                    dst = self._zhipu.translate([t for _, t in pending], target,
                                                image_path=image_path)
                except Exception as e:
                    self.log(f"[翻译] 智谱看图翻译失败，走通用链路：{e}")
                    dst = None
                if dst is not None:
                    for (i, _), d in zip(pending, dst):
                        results[i] = d
            if any(r is None for r in results):
                rest = [(i, t) for i, t in pending if results[i] is None]
                dst = self._translate_batch([t for _, t in rest], target)
                for (i, _), d in zip(rest, dst):
                    results[i] = d
        if self._zhipu is not None:
            results = self._zhipu._hygiene([_post_fix(r, target) for r in results], target)
            return results
        return [_post_fix(r, target) for r in results]

    def translate(self, texts, target, batch_size=20):
        """texts: 中文文本列表；target: 'ru'/'es'/'pt'。返回等长译文列表。
        命中电商术语库的短语直接使用精译。"""
        assert target in SUPPORTED_TARGETS, f"不支持的目标语言: {target}"
        from .glossary import lookup_glossary
        results = [None] * len(texts)
        pending = []
        for i, t in enumerate(texts):
            hit = lookup_glossary(t, target)
            if hit:
                results[i] = hit
            else:
                pending.append((i, _pre_fix(t)))
        for j in range(0, len(pending), batch_size):
            chunk = pending[j:j + batch_size]
            translated = self._translate_batch([t for _, t in chunk], target)
            for (i, _), dst in zip(chunk, translated):
                results[i] = dst
        return [_post_fix(r, target) for r in results]


def _pre_fix(text):
    """源文预处理：把易误译的单位缩写写成中文全称，提升机翻正确率。"""
    import re
    # "30oz" -> "30盎司"，避免被译成操作系统等离谱结果
    # 兼容 OCR 把字母 o 误识为数字 0（"300z" -> "30盎司"）
    text = re.sub(r"(\d)\s*[oO0][zZ]\b", r"\1盎司", text)
    return text


def _post_fix(text, target):
    """译文后处理：单位等机翻常见错误的修正。"""
    if not text:
        return text
    import re
    if target == "ru":
        text = re.sub(r"(\d)\s*[оО]с\b", r"\1 унций", text)
    elif target == "pt":
        text = re.sub(r"(\d)\s*sistemas operacionais\b", r"\1 onças", text,
                      flags=re.IGNORECASE)
    elif target == "es":
        text = re.sub(r"(\d)\s*sistemas operativos\b", r"\1 onzas", text,
                      flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    engine = TranslateEngine()
    samples = ["你好，世界！", "这款软件可以批量翻译图片和视频。", "价格实惠，质量保证，欢迎选购。"]
    for lang in ("ru", "es", "pt"):
        print(f"--- {SUPPORTED_TARGETS[lang]} ---")
        for src, dst in zip(samples, engine.translate(samples, lang)):
            print(f"{src}  ->  {dst}")
