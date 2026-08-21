# ============================================================
# app.py – V-SPLADE + SparseVec + PyWebView
# ============================================================
import os
import sys
import gc
import json
import time
import logging
import threading
import traceback
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Palette images")

import io
import base64
import numpy as np
import requests
import torch
import webview
from PIL import Image
from tqdm import tqdm
from scipy import sparse as sp

# ── v-splade 경로 설정 ──────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

VSPLADE_DIR = APP_DIR / "v-splade"
VSPLADE_EXAMPLES = VSPLADE_DIR / "examples"
if str(VSPLADE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(VSPLADE_EXAMPLES))

# ── 모델 파일 다운로드 경로 ───────────────────────────────────
_local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
if not _local_app_data:
    if os.name == "nt":
        _local_app_data = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        _local_app_data = os.path.join(os.path.expanduser("~"), ".local", "share")
LOCAL_APP_DATA = os.path.join(_local_app_data, "VSpladeSearch")

VSPLADE_MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
VSPLADE_REPO = "naver/v-splade-quality"
VSPLADE_URLS = {
    fname: f"https://huggingface.co/{VSPLADE_REPO}/resolve/main/{fname}"
    for fname in VSPLADE_MODEL_FILES
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif", ".pdf"}
PDF_EXTS = {".pdf"}

# ── 번역 모델 정의 ──────────────────────────────────────────
import locale

TRANSLATION_MODEL_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "source.spm", "special_tokens_map.json", "target.spm",
    "tokenizer_config.json", "vocab.json", "vocab.spm", "added_tokens.json",
]

TRANSLATION_LANGS = {
    "kor": {"name": "한국어",    "model": "kor-eng"},
    "fra": {"name": "Français",  "model": "fra-eng"},
    "deu": {"name": "Deutsch",   "model": "deu-eng"},
    "ita": {"name": "Italiano",  "model": "ita-eng"},
    "nld": {"name": "Nederlands","model": "nld-eng"},
    "rus": {"name": "Русский",   "model": "rus-eng"},
    "ara": {"name": "العربية",   "model": "ara-eng"},
    "zho": {"name": "中文",      "model": "zho-eng"},
    "ell": {"name": "Ελληνικά",  "model": "ell-eng"},
    "tur": {"name": "Türkçe",    "model": "tur-eng"},
    "spa": {"name": "Español",   "model": "spa-eng"},
    "cat": {"name": "Català",    "model": "cat-eng"},
    "eng": {"name": "English (원본)", "model": None},
}

TRANSLATION_DOWNLOAD_URLS = {}
for _code, _info in TRANSLATION_LANGS.items():
    if _info["model"] is not None:
        _base = f"https://huggingface.co/Helsinki-NLP/opus-mt_tiny_{_info['model']}/resolve/main"
        TRANSLATION_DOWNLOAD_URLS[_info["model"]] = {
            fname: f"{_base}/{fname}" for fname in TRANSLATION_MODEL_FILES
        }

def _detect_pc_language() -> str:
    try:
        lang_code = locale.getlocale()[0]
        if lang_code:
            short = lang_code.split("_")[0].lower()
            if short in TRANSLATION_LANGS:
                return short
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            _map = {
                0x0412: "kor", 0x040C: "fra", 0x0407: "deu",
                0x0410: "ita", 0x0413: "nld", 0x0419: "rus",
                0x0401: "ara", 0x0804: "zho", 0x0408: "ell",
                0x041F: "tur", 0x0C0A: "spa", 0x040A: "spa",
                0x0403: "cat", 0x0409: "eng",
            }
            if lang_id in _map:
                return _map[lang_id]
        except Exception:
            pass
    return "eng"

PC_LANGUAGE = _detect_pc_language()

# ── 로깅 설정 ────────────────────────────────────────────────
os.makedirs(LOCAL_APP_DATA, exist_ok=True)
LOG_FILE = os.path.join(LOCAL_APP_DATA, "app.log")

logger = logging.getLogger("VSpladeSearch")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(funcName)s - %(message)s"))

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(_fh)
logger.addHandler(_ch)

logger.info("=== VSpladeSearch 앱 시작 (V-SPLADE) ===")
logger.info("APP_DIR: %s", APP_DIR)
logger.info("LOCAL_APP_DATA: %s", LOCAL_APP_DATA)
logger.info("LOG_FILE: %s", LOG_FILE)

# ── 만료된 HF 토큰 파일 제거 ────────────────────────────────
_hf_token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
if os.path.isfile(_hf_token_path):
    try:
        os.remove(_hf_token_path)
        logger.info("[HF] 만료된 토큰 파일 제거: %s", _hf_token_path)
    except Exception as e:
        logger.warning("[HF] 토큰 파일 제거 실패: %s", e)


# ============================================================
#  SparseVec – scipy CSR 기반 sparse 벡터 DB
# ============================================================
class SparseVec:
    """Sparse 벡터 저장소 (dot product 기반)."""

    def __init__(self, vocab_size: int = 50368):
        self.vocab_size = vocab_size
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self._rows: list = []
        self.matrix = None

    def add(self, vec_id: str, vec, metadata: dict | None = None):
        """vec: torch.Tensor (vocab_size,) 또는 scipy sparse row."""
        if isinstance(vec, torch.Tensor):
            v = vec.float().cpu()
            nz = v.nonzero(as_tuple=False).squeeze(-1).numpy().astype(np.int64)
            vals = v[nz].numpy().astype(np.float32)
            row = sp.csr_matrix(
                (vals, (np.zeros_like(nz), nz)),
                shape=(1, self.vocab_size),
            )
        elif sp.issparse(vec):
            row = vec.tocsr()
        else:
            raise ValueError(f"Unsupported vec type: {type(vec)}")

        self._rows.append(row)
        self.ids.append(vec_id)
        self.meta.append(metadata or {})

    def _build_matrix(self):
        if self.matrix is None and self._rows:
            self.matrix = sp.vstack(self._rows, format="csr")

    def search(self, query_vec, top_k: int = 10, return_zero: bool = True):
        if not self.ids:
            return []

        if isinstance(query_vec, torch.Tensor):
            v = query_vec.detach().float().cpu()
            if v.ndim == 2:
                v = v.squeeze(0)
            nz = v.nonzero(as_tuple=False).squeeze(-1).numpy().astype(np.int64)
            vals = v[nz].numpy().astype(np.float32)

            q = sp.csr_matrix(
                (vals, (np.zeros_like(nz), nz)),
                shape=(1, self.vocab_size),
            )

        elif sp.issparse(query_vec):
            q = query_vec.tocsr()

        else:
            v = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            nz = np.nonzero(v)[0].astype(np.int64)
            vals = v[nz].astype(np.float32)
            q = sp.csr_matrix(
                (vals, (np.zeros_like(nz), nz)),
                shape=(1, self.vocab_size),
            )

        self._build_matrix()

        scores = self.matrix.dot(q.T).toarray().ravel()

        # NaN 방지
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            if scores[i] > 0.0:
                results.append({
                    "id": self.ids[i],
                    "score": float(scores[i]),
                    **self.meta[i],
                })

        return results

    def save(self, path: str):
        self._build_matrix()
        npz_path = path
        json_path = path.replace(".npz", ".json")
        if self.matrix is not None:
            sp.save_npz(npz_path, self.matrix)
        else:
            sp.save_npz(npz_path, sp.csr_matrix((0, self.vocab_size)))
        data = {
            "vocab_size": self.vocab_size,
            "ids": self.ids,
            "meta": self.meta,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        npz_path = path
        json_path = path.replace(".npz", ".json")
        self.matrix = sp.load_npz(npz_path)
        self.vocab_size = self.matrix.shape[1]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.meta = data["meta"]
        self._rows = []

    def __len__(self):
        return len(self.ids)

# ============================================================
#  모델 다운로드 & 로드
# ============================================================
def _detect_accelerator():
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return torch.device("cuda"), "ROCm (AMD GPU)"
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    return torch.device("cpu"), "CPU"


def _download_file(url: str, dest: str, desc: str = ""):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dest)
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def load_vsplade_model(model_dir: str):
    from vsplade_inference import VSPLADEInference

    device, accel_name = _detect_accelerator()
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = torch.float32
    logger.info("[모델] V-SPLADE 로드 중: %s (device=%s, dtype=%s)", model_dir, device, dtype)
    model = VSPLADEInference.from_pretrained(model_dir, device=str(device), dtype=dtype)

    # ── chat_template 주입: 가능한 모든 경로 시도 ──
    _CHAT_TMPL = (
        "{% for message in messages %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}<image>"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endfor %}"
    )
    _injected = False

    # 1) model.processor 직접 접근
    _proc = getattr(model, 'processor', None)
    # 2) model._processor
    if _proc is None:
        _proc = getattr(model, '_processor', None)
    # 3) model.collator.processor
    if _proc is None:
        _col = getattr(model, 'collator', None)
        if _col is not None:
            _proc = getattr(_col, 'processor', None)
    # 4) model.model 내부 순회
    if _proc is None:
        for attr_name in dir(model):
            obj = getattr(model, attr_name, None)
            if obj is not None and hasattr(obj, 'process_images'):
                _proc = obj
                break

    if _proc is not None:
        # processor 자체에 설정 (가장 중요)
        try:
            _proc.chat_template = _CHAT_TMPL
        except Exception:
            pass
        # tokenizer에도 설정
        _tok = getattr(_proc, 'tokenizer', None)
        if _tok is not None:
            _tok.chat_template = _CHAT_TMPL
        _injected = True
        logger.info("[모델] chat_template 주입 완료 (processor 경로: %s)", type(_proc).__name__)
    else:
        logger.warning("[모델] processor를 찾지 못함! chat_template 주입 실패")

    # 5) 최종 폴백: tokenizer 직접 접근
    if not _injected:
        _tok = getattr(model, 'tokenizer', None)
        if _tok is not None:
            _tok.chat_template = _CHAT_TMPL
            logger.info("[모델] chat_template 주입 (tokenizer 직접)")

    logger.info("[모델] V-SPLADE 로드 완료 (vocab_size=%d)", model.model.vocab_size)
    return model, device, dtype

# ============================================================
#  PyWebView ↔ Python 브리지 (Api)
# ============================================================
class Api:
    def __init__(self):
        self.model = None
        self.device = None
        self.dtype = None
        self.svec_db = SparseVec(vocab_size=50368)
        self.indexed_count = 0
        self.total_images = 0
        self.progress_msg = ""
        self._indexing = False
        self._downloading = False
        self._download_pct = 0.0
        self._download_msg = ""
        self._model_ready = False
        self.recent_indexed: list[dict] = []
        self.translator_model = None
        self.translator_tokenizer = None
        self.selected_lang = PC_LANGUAGE
        self.last_indexed_folder = ""
        self._index_path = os.path.join(LOCAL_APP_DATA, "sparse_index.npz")
        self._index_meta_path = os.path.join(LOCAL_APP_DATA, "sparse_index.json")
        self._index_state_path = os.path.join(LOCAL_APP_DATA, "index_state.json")
        self._model_dir = os.path.join(LOCAL_APP_DATA, "vsplade-quality")
        logger.info("[Api] 초기화 완료 (PC 언어: %s)", PC_LANGUAGE)



    # ── 인덱스 상태 저장/복원 ────────────────────────────────
    def _save_index_state(self, folder: str):
        try:
            state = {
                "folder": folder,
                "count": len(self.svec_db),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self._index_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            self.last_indexed_folder = folder
        except Exception as e:
            logger.error("[상태] 저장 실패: %s", e)

    def _try_auto_load_index(self):
        try:
            if not (os.path.isfile(self._index_path) and os.path.isfile(self._index_meta_path)):
                return {"loaded": False, "msg": "저장된 인덱스 없음"}
            self.svec_db.load(self._index_path)
            folder = ""
            if os.path.isfile(self._index_state_path):
                with open(self._index_state_path, "r", encoding="utf-8") as f:
                    folder = json.load(f).get("folder", "")
            self.last_indexed_folder = folder
            self.indexed_count = len(self.svec_db)
            self.total_images = len(self.svec_db)
            self.recent_indexed = []
            for i in range(max(0, len(self.svec_db) - 30), len(self.svec_db)):
                meta = self.svec_db.meta[i]
                p = meta.get("path", "")
                self.recent_indexed.append({"path": p, "name": meta.get("name", Path(p).name if p else "")})
            logger.info("[자동로드] 완료: %d개, 폴더: %s", len(self.svec_db), folder)
            return {"loaded": True, "count": len(self.svec_db), "folder": folder}
        except Exception as e:
            logger.error("[자동로드] 실패: %s", e, exc_info=True)
            return {"loaded": False, "msg": str(e)}

    # ── 언어 리스트 ──────────────────────────────────────────
    def get_language_list(self):
        langs = [{"code": c, "name": i["name"]} for c, i in TRANSLATION_LANGS.items()]
        return {"ok": True, "langs": langs, "default": PC_LANGUAGE}

    # ── 번역 모델 ────────────────────────────────────────────
    def load_translator(self, lang_code: str):
        logger.info("[이벤트] 번역 모델 로드 요청: %s", lang_code)
        block = self._check_ready()
        if block:
            return block
        if lang_code == "eng":
            self.selected_lang = "eng"
            self.translator_model = None
            self.translator_tokenizer = None
            return {"ok": True, "msg": "영어 선택됨 (번역 불필요)"}
        info = TRANSLATION_LANGS.get(lang_code)
        if not info or info["model"] is None:
            return {"ok": False, "msg": f"지원하지 않는 언어: {lang_code}"}
        model_pair = info["model"]
        local_dir = os.path.join(LOCAL_APP_DATA, "translation", model_pair)
        os.makedirs(local_dir, exist_ok=True)
        missing = []
        for fname in TRANSLATION_MODEL_FILES:
            fpath = os.path.join(local_dir, fname)
            if not os.path.isfile(fpath):
                missing.append((fname, TRANSLATION_DOWNLOAD_URLS[model_pair][fname], fpath))
        if missing:
            threading.Thread(target=self._download_translator_files, args=(missing, local_dir, model_pair, lang_code), daemon=True).start()
            return {"ok": True, "msg": f"번역 모델 다운로드 시작 ({len(missing)}개 파일)"}
        return self._load_translator_model(local_dir, lang_code, model_pair)

    def _make_thumb_b64_from_image(self, img: Image.Image, size: int = 220) -> str:
        """PIL Image로부터 직접 썸네일 생성 (PDF 페이지용)."""
        try:
            thumb = img.copy()
            thumb.thumbnail((size, size), Image.LANCZOS)
            if thumb.mode != "RGB":
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=80)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        except Exception:
            return ""

    def _download_translator_files(self, missing, local_dir, model_pair, lang_code):
        try:
            for fname, url, fpath in missing:
                self._download_msg = f"번역 모델 다운로드: {fname}"
                self._download_file_with_progress(url, fpath, fname)
            self._load_translator_model(local_dir, lang_code, model_pair)
        except Exception as e:
            logger.error("[번역] 다운로드 실패: %s", e, exc_info=True)

    def _load_translator_model(self, local_dir, lang_code, model_pair):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.translator_tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
            self.translator_model.eval()
            self.translator_model = self.translator_model.to(self.device)
            self.selected_lang = lang_code
            logger.info("[번역] 모델 로드 완료: %s", model_pair)
            return {"ok": True, "msg": f"번역 모델 로드 완료 ({model_pair})"}
        except Exception as e:
            logger.error("[번역] 모델 로드 실패: %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    def _has_target_lang(self, text: str, lang_code: str) -> bool:
        if lang_code == "kor":
            return any('\uac00' <= ch <= '\ud7a3' or '\u1100' <= ch <= '\u11ff' for ch in text)
        elif lang_code == "zho":
            return any('\u4e00' <= ch <= '\u9fff' for ch in text)
        elif lang_code == "ara":
            return any('\u0600' <= ch <= '\u06ff' for ch in text)
        elif lang_code == "ell":
            return any('\u0370' <= ch <= '\u03ff' for ch in text)
        elif lang_code == "rus":
            return any('\u0400' <= ch <= '\u04ff' for ch in text)
        return True

    def _translate_to_english(self, text: str) -> str:
        if self.translator_model is None or self.translator_tokenizer is None:
            return text
        try:
            inputs = self.translator_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.inference_mode():
                outputs = self.translator_model.generate(**inputs, max_length=128, num_beams=4)
            return self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning("[번역] 실패: %s", e)
            return text

    def _auto_load_translator(self, lang_code: str):
        try:
            info = TRANSLATION_LANGS.get(lang_code)
            if not info or info["model"] is None:
                return
            model_pair = info["model"]
            local_dir = os.path.join(LOCAL_APP_DATA, "translation", model_pair)
            os.makedirs(local_dir, exist_ok=True)
            missing = []
            for fname in TRANSLATION_MODEL_FILES:
                fpath = os.path.join(local_dir, fname)
                if not os.path.isfile(fpath):
                    missing.append((fname, TRANSLATION_DOWNLOAD_URLS[model_pair][fname], fpath))
            if missing:
                for fname, url, fpath in missing:
                    self._download_msg = f"번역 모델 다운로드: {fname}"
                    self._download_file_with_progress(url, fpath, fname)
            self._load_translator_model(local_dir, lang_code, model_pair)
        except Exception as e:
            logger.error("[번역] 자동 로드 실패: %s", e, exc_info=True)

    # ── 모델 준비 체크 ───────────────────────────────────────
    def _check_ready(self):
        if self._downloading:
            return {"ok": False, "msg": "모델 다운로드 중입니다."}
        if not self._model_ready:
            return {"ok": False, "msg": "모델이 준비되지 않았습니다."}
        return None

    # ── 초기화 ───────────────────────────────────────────────
    def init_model(self):
        logger.info("[이벤트] 모델 다운로드/초기화 요청")
        if self._downloading:
            return {"ok": False, "msg": "이미 다운로드 중입니다."}
        if self._model_ready:
            return {"ok": True, "msg": "모델 이미 로드됨"}
        self._downloading = True
        self._download_pct = 0.0
        self._download_msg = "다운로드 준비 중..."
        threading.Thread(target=self._init_worker, daemon=True).start()
        return {"ok": True, "msg": "다운로드 시작"}

    def _init_worker(self):
        try:
            model_dir = self._ensure_model_files_with_progress()
            self._download_msg = "V-SPLADE 모델 로드 중..."
            self._download_pct = 90.0
            self.model, self.device, self.dtype = load_vsplade_model(model_dir)

            self._download_msg = "이전 인덱스 확인 중..."
            self._download_pct = 93.0
            auto = self._try_auto_load_index()

            if PC_LANGUAGE != "eng":
                self._download_msg = f"번역 모델 로드 중 ({PC_LANGUAGE})..."
                self._download_pct = 96.0
                self._auto_load_translator(PC_LANGUAGE)

            if auto.get("loaded"):
                self._download_msg = f"모델 로드 완료! 이전 인덱스 {auto['count']}개 복원됨"
            else:
                self._download_msg = "모델 로드 완료!"
            self._model_ready = True
            self._download_pct = 100.0
        except Exception as e:
            logger.error("[모델 초기화 오류] %s", e, exc_info=True)
            self._download_msg = f"오류: {e}"
        finally:
            self._downloading = False

    def _ensure_model_files_with_progress(self):
        os.makedirs(self._model_dir, exist_ok=True)
        for fname in VSPLADE_MODEL_FILES:
            fpath = os.path.join(self._model_dir, fname)
            if not os.path.isfile(fpath):
                self._download_msg = f"V-SPLADE 다운로드: {fname}"
                self._download_file_with_progress(VSPLADE_URLS[fname], fpath, fname)
        return self._model_dir

    def _download_file_with_progress(self, url, dest, desc):
        logger.info("[다운로드] %s 시작", desc)
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    self._download_pct = round(downloaded / total * 100, 1)
                mb_dl = downloaded // (1 << 20)
                mb_total = total // (1 << 20)
                self._download_msg = f"{desc} {self._download_pct}% ({mb_dl}MB / {mb_total}MB)"
        logger.info("[다운로드] %s 완료", desc)

    def get_download_progress(self):
        return {
            "downloading": self._downloading,
            "pct": self._download_pct,
            "msg": self._download_msg,
            "ready": self._model_ready,
        }

    # ── 폴더 선택 / 스캔 ─────────────────────────────────────
    def select_folder(self):
        logger.info("[이벤트] 폴더 선택 버튼 클릭")
        block = self._check_ready()
        if block:
            return block
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="이미지 폴더 선택")
            root.destroy()
            if not folder:
                return {"ok": False, "msg": "폴더가 선택되지 않았습니다."}
            logger.info("[폴더 선택] %s", folder)
            return {"ok": True, "path": folder}
        except Exception as e:
            logger.error("[폴더 선택 오류] %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    def scan_images(self, folder_path: str):
        images = []
        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    images.append(os.path.join(root, fname))
        return {"ok": True, "count": len(images), "images": images}

    # ── 인덱싱 ───────────────────────────────────────────────
    def start_indexing(self, folder_path: str):
        logger.info("[이벤트] 인덱싱 요청: %s", folder_path)
        block = self._check_ready()
        if block:
            return block
        if self._indexing:
            return {"ok": False, "msg": "이미 인덱싱 중입니다."}
        self._indexing = True
        threading.Thread(target=self._index_worker, args=(folder_path,), daemon=True).start()
        return {"ok": True, "msg": "인덱싱 시작"}

    def reset_index(self):
        """인덱스 전체 초기화 (DB + 디스크 파일 + 갤러리)."""
        block = self._check_ready()
        if block:
            return block
        if self._indexing:
            return {"ok": False, "msg": "인덱싱 중에는 초기화할 수 없습니다."}
        self.svec_db = SparseVec(vocab_size=self.model.model.vocab_size)
        self.recent_indexed = []
        self.indexed_count = 0
        self.total_images = 0
        self.last_indexed_folder = ""
        for f in (self._index_path, self._index_meta_path, self._index_state_path):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except Exception as e:
                logger.warning("[리셋] 파일 삭제 실패: %s (%s)", f, e)
        logger.info("[리셋] 인덱스가 초기화되었습니다.")
        return {"ok": True, "msg": "인덱스가 초기화되었습니다."}

    def _index_worker(self, folder_path: str):
        logger.info("[인덱싱 워커] 시작: %s", folder_path)
        try:
            images = []
            for root, _dirs, files in os.walk(folder_path):
                for fname in files:
                    if Path(fname).suffix.lower() in IMAGE_EXTS:
                        images.append(os.path.join(root, fname))

            # ── 사전 스캔: PDF 페이지 수까지 포함 ──
            total_items = 0
            pdf_page_counts = {}
            for p in images:
                if Path(p).suffix.lower() in PDF_EXTS:
                    try:
                        import fitz
                        doc = fitz.open(p)
                        page_count = min(len(doc), 10)
                        doc.close()
                    except Exception:
                        page_count = 0
                    pdf_page_counts[p] = page_count
                    total_items += page_count
                else:
                    total_items += 1

            self.total_images = total_items
            self.indexed_count = 0
            self.svec_db = SparseVec(vocab_size=self.model.model.vocab_size)
            self.recent_indexed = []   # ← 추가: 재인덱싱 시 갤러리 중복 방지
            logger.info("[인덱싱] 총 %d개 항목 (파일 %d개, PDF 페이지 포함), vocab_size=%d",
                        self.total_images, len(images), self.svec_db.vocab_size)

            diag_count = 0  # 진단 로그는 앞의 5개 문서만 출력
            BATCH = int(os.environ.get("VSPLADE_BATCH", "4"))
            pending: list = []

            def _flush_pending():
                nonlocal diag_count
                if not pending:
                    return
                pil_imgs = [it["pil"] for it in pending]
                vecs = self._encode_images_batch(pil_imgs)
                for it, vec in zip(pending, vecs):
                    nnz = int((vec > 0).sum())
                    if nnz == 0:
                        logger.warning("[인덱싱] ⚠️ 빈 벡터: %s (nnz=0)", it["name"])
                    elif diag_count < 5:
                        top = self.model.decode_topk(vec, k=5)
                        logger.info("[인덱싱] %s nnz=%d | %s", it["name"], nnz,
                                    ", ".join(f"'{t}'({w:.2f})" for t, w in top))
                        diag_count += 1
                    self.svec_db.add(it["vec_id"], vec, it["meta"])
                    self.indexed_count += 1
                    thumb = (self._make_thumb_b64_from_image(it["pil"])
                             if it["is_pdf"] else self._make_thumb_b64(it["path"]))
                    self.recent_indexed.append({
                        "path": it["path"], "name": it["name"], "thumb_b64": thumb,
                    })
                    self.progress_msg = f"{self.indexed_count}/{self.total_images} 인덱싱 완료"
                pending.clear()
                if len(self.recent_indexed) > 30:
                    self.recent_indexed = self.recent_indexed[-30:]
                logger.info("[인덱싱] %s", self.progress_msg)

            for idx, p in enumerate(images):
                try:
                    if Path(p).suffix.lower() in PDF_EXTS:
                        pdf_imgs = self._pdf_to_images(p)
                        for page_idx, pi in enumerate(pdf_imgs):
                            display_name = f"{Path(p).name} (p.{page_idx + 1}/{len(pdf_imgs)})"
                            pending.append({
                                "pil": pi, "is_pdf": True, "path": p,
                                "vec_id": f"{p}#p{page_idx + 1}",
                                "name": display_name,
                                "meta": {"path": p, "name": display_name,
                                         "page": page_idx + 1,
                                         "total_pages": len(pdf_imgs)},
                            })
                            if len(pending) >= BATCH:
                                _flush_pending()
                    else:
                        img = Image.open(p)
                        if img.mode in ("RGBA", "LA", "P"):
                            img = img.convert("RGBA")
                            bg = Image.new("RGB", img.size, (255, 255, 255))
                            bg.paste(img, mask=img.split()[-1])
                            img = bg
                        else:
                            img = img.convert("RGB")
                        pending.append({
                            "pil": img, "is_pdf": False, "path": p,
                            "vec_id": p, "name": Path(p).name,
                            "meta": {"path": p, "name": Path(p).name},
                        })
                        if len(pending) >= BATCH:
                            _flush_pending()
                except Exception as img_err:
                    logger.warning("[인덱싱] 로드 실패: %s (%s)", p, img_err)
                    if Path(p).suffix.lower() in PDF_EXTS:
                        self.indexed_count += pdf_page_counts.get(p, 0)
                    else:
                        self.indexed_count += 1
                    continue
            _flush_pending()

            self.svec_db.save(self._index_path)
            self._save_index_state(folder_path)
            self.progress_msg = f"인덱싱 완료! 총 {self.indexed_count}개. 저장: {self._index_path}"
            logger.info("[인덱싱] %s", self.progress_msg)
        except Exception as e:
            logger.error("[인덱싱 오류] %s", e, exc_info=True)
            self.progress_msg = f"인덱싱 오류: {e}"
        finally:
            self._indexing = False
            logger.info("[인덱싱 워커] 종료")

    def _encode_images_batch(self, pil_imgs: list):
        """이미지 여러 장을 1회 forward로 인코딩. 실패 시 건별 인코딩으로 폴백."""
        try:
            prompts = []
            for _ in pil_imgs:
                chat = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
                prompts.append(
                    self.model.processor.apply_chat_template(chat, add_generation_prompt=True)
                )
            inputs = self.model.processor(
                images=[im.convert("RGB") for im in pil_imgs],
                text=prompts,
                return_tensors="pt",
            )
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with torch.inference_mode():
                w = self.model.model.encode_passage(**inputs)
            if isinstance(w, (tuple, list)):
                w = w[0]
            return [w[i] for i in range(w.shape[0])]
        except Exception as e:
            logger.warning("[인덱싱] 배치 인코딩 실패 → 건별 폴백: %s", e)
            return [self.model.encode_image(im) for im in pil_imgs]

    def get_progress(self):
        recent_out = []
        for r in reversed(self.recent_indexed):
            item = dict(r)
            if not item.get("thumb_b64") and item.get("path"):
                item["thumb_b64"] = self._make_thumb_b64(item["path"])
                r["thumb_b64"] = item["thumb_b64"]
            recent_out.append(item)
        return {
            "indexing": self._indexing,
            "current": self.indexed_count,
            "total": self.total_images,
            "msg": self.progress_msg,
            "recent": recent_out,
        }

    # ── 검색 ─────────────────────────────────────────────────
    # ── 전체 목록 조회 ─────────────────────────────────────
    def list_all(self):
        logger.info("[이벤트] 전체 목록 요청")
        block = self._check_ready()
        if block:
            return block
        if len(self.svec_db) == 0:
            return {"ok": False, "msg": "인덱스가 비어 있습니다."}
        results = []
        for i in range(len(self.svec_db)):
            meta = self.svec_db.meta[i]
            results.append({
                "id": self.svec_db.ids[i],
                "path": meta.get("path", ""),
                "name": meta.get("name", ""),
            })
        for r in results:
            r["thumb_b64"] = self._make_thumb_b64(r.get("path", ""))
        logger.info("[전체 목록] %d개 반환", len(results))
        return {"ok": True, "results": results}

    def search(self, query: str, top_k: int = 20):
        logger.info("[이벤트] 검색: '%s' (top_k=%d)", query, top_k)
        block = self._check_ready()
        if block:
            return block
        if len(self.svec_db) == 0:
            return {"ok": False, "msg": "인덱스가 비어 있습니다."}
        try:
            translated = query
            if self.selected_lang != "eng" and self._has_target_lang(query, self.selected_lang):
                translated = self._translate_to_english(query)
                logger.info("[번역] '%s' → '%s'", query, translated)

            q_vec = self.model.encode_query(translated)

            # ── 쿼리 토큰 로깅 ──
            try:
                q_tokens = self.model.decode_topk(q_vec, k=15)
                token_str = ", ".join(f"'{t}'({w:.3f})" for t, w in q_tokens)
                logger.info("[쿼리 토큰] %s", token_str)
            except Exception as e:
                logger.debug("[쿼리 토큰] 디코딩 실패: %s", e)

            results = self.svec_db.search(q_vec, top_k=top_k)

            # ── 검색 결과 상세 로깅 ──
            logger.info("[검색 결과] query='%s' (translated='%s') | 총 %d개 결과",
                        query, translated, len(results))
            for i, r in enumerate(results):
                logger.info("  #%02d  score=%.4f  %s",
                            i + 1, r["score"], r.get("name", r.get("id", "?")))

            for r in results:
                r["thumb_b64"] = self._make_thumb_b64(r.get("path", ""))


            return {"ok": True, "results": results, "translated": translated}
        except Exception as e:
            logger.error("[검색 오류] %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    # ── PDF / 썸네일 ─────────────────────────────────────────
    def _pdf_to_images(self, pdf_path: str, max_pages: int = 10, dpi: int = 150) -> list:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            images = []
            for i in range(min(len(doc), max_pages)):
                pix = doc[i].get_pixmap(dpi=dpi)
                images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            doc.close()
            return images
        except Exception as e:
            logger.warning("[PDF] 변환 실패: %s (%s)", pdf_path, e)
            return []

    def _make_thumb_b64(self, path: str, size: int = 220) -> str:
        try:
            if Path(path).suffix.lower() in PDF_EXTS:
                imgs = self._pdf_to_images(path, max_pages=1, dpi=100)
                if not imgs:
                    return ""
                img = imgs[0]
            else:
                img = Image.open(path)
            img.thumbnail((size, size), Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        except Exception as e:
            return ""

    # ── 인덱스 상태/로드 ─────────────────────────────────────
    def get_index_state(self):
        return {
            "has_index": len(self.svec_db) > 0,
            "count": len(self.svec_db),
            "folder": self.last_indexed_folder,
            "index_path": self._index_path,
        }

    def load_index(self):
        block = self._check_ready()
        if block:
            return block
        if os.path.isfile(self._index_path) and os.path.isfile(self._index_meta_path):
            self.svec_db.load(self._index_path)
            return {"ok": True, "msg": f"인덱스 로드 완료 ({len(self.svec_db)}개)"}
        return {"ok": False, "msg": "저장된 인덱스가 없습니다."}

# ============================================================
#  HTML / JS – PyWebView 렌더링
# ============================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<title>XY Zvec - V Splade Search</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    background: #0f1117; color: #e0e0e0;
    display:flex; flex-direction:column; height:100vh;
  }
  header {
    background:#1a1d27; padding:16px 24px;
    display:flex; align-items:center; gap:16px;
    border-bottom:1px solid #2a2d3a;
  }
  header h1 { font-size:20px; color:#7eb8ff; }
  .toolbar {
    padding:12px 24px; display:flex; gap:12px;
    align-items:center; flex-wrap:wrap;
    background:#14161f;
  }
  button {
    background:#2563eb; color:#fff; border:none;
    padding:10px 20px; border-radius:8px; cursor:pointer;
    font-size:14px; transition:background .2s;
  }
  button:hover { background:#1d4ed8; }
  button:disabled { background:#3a3d4a; cursor:not-allowed; }
  .search-box {
    flex:1; display:flex; gap:8px; min-width:280px;
  }
  .search-box input {
    flex:1; padding:10px 14px; border-radius:8px;
    border:1px solid #3a3d4a; background:#1e2130;
    color:#fff; font-size:14px; outline:none;
  }
  .search-box input:focus { border-color:#2563eb; }
  #status {
    padding:8px 24px; font-size:13px; color:#9ca3af;
    background:#14161f; border-bottom:1px solid #1e2130;
  }
  .progress-bar {
    height:4px; background:#1e2130; border-radius:2px;
    margin:6px 24px; overflow:hidden;
  }
  .progress-bar .fill {
    height:100%; background:#2563eb; width:0%;
    transition:width .3s;
  }
  #gallery {
    flex:1; overflow-y:auto; padding:16px 24px;
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(200px,1fr));
    gap:14px; align-content:start;
  }
  .card {
    background:#1a1d27; border-radius:10px;
    border:1px solid #2a2d3a; transition:transform .15s;
  }
  .card:hover { transform:translateY(-3px); }
  .card img {
    width:100%; height:160px; object-fit:cover; display:block;
  }
  .card .info {
    padding:8px 10px; font-size:12px; color:#9ca3af;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .card .score {
    color:#7eb8ff; font-weight:600;
  }
  .empty-msg {
    grid-column:1/-1; text-align:center;
    color:#555; padding:60px 0; font-size:15px;
  }
  .card img {
    cursor:zoom-in;
  }
  #imgModal {
    display:none;
    position:fixed;
    top:0; left:0;
    width:100vw; height:100vh;
    background:rgba(0,0,0,0.85);
    z-index:9999;
    align-items:center;
    justify-content:center;
    cursor:pointer;
  }
  #imgModalInner {
    position:relative;
    max-width:90vw;
    max-height:90vh;
    text-align:center;
  }
  #imgModalClose {
    position:absolute;
    top:-44px; right:0;
    background:#ef4444;
    color:#fff;
    border:none;
    border-radius:8px;
    padding:8px 18px;
    cursor:pointer;
    font-size:14px;
    z-index:10000;
  }
  #imgModalClose:hover {
    background:#dc2626;
  }
  #imgModalImg {
    max-width:90vw;
    max-height:82vh;
    object-fit:contain;
    border-radius:8px;
    display:block;
    margin:0 auto;
  }
  #imgModalInfo {
    color:#e0e0e0;
    padding:10px 0 4px 0;
    font-size:13px;
  }
</style>
</head>
<body>

<header>
  <h1>🔍 XY Zvec - V Splade Search</h1>
  <span style="font-size:13px;color:#666;">
    자연어로 문서를 검색하세요
  </span>
</header>

<div class="toolbar">
  <button id="btnDownload" onclick="startDownload()"
          style="display:none;background:#16a34a;">⬇️ 모델 다운로드</button>
  <button id="btnFolder" onclick="selectFolder()" disabled>📁 폴더 선택</button>
  <button id="btnIndex" onclick="startIndexing()" disabled>⚙️ 인덱싱 시작</button>
  <button id="btnLoadIdx" onclick="loadIndex()" disabled>📂 저장된 인덱스 로드</button>
  <button id="btnReset" onclick="resetIndex()" disabled style="background:#dc2626;">🗑️ 인덱스 초기화</button>
  <select id="langSelect" onchange="changeLanguage()" disabled
          style="padding:10px 14px;border-radius:8px;border:1px solid #3a3d4a;
                 background:#1e2130;color:#fff;font-size:14px;outline:none;
                 cursor:pointer;">
  </select>
  <div class="search-box">
    <button id="btnClear" title="검색어 지우기"
            style="display:none;background:#3a3d4a;padding:10px 14px;"
            onclick="var i=document.getElementById('searchInput');i.value='';this.style.display='none';i.focus();showAll();">✕</button>
    <input id="searchInput" type="text" disabled
           placeholder="자연어로 문서 검색… (빈칸 엔터 = 전체 목록)"
           oninput="document.getElementById('btnClear').style.display=this.value?'inline-block':'none'"
           onkeydown="if(event.key==='Enter')doSearch()"/>
    <button id="btnSearch" onclick="doSearch()" disabled>검색</button>
  </div>
</div>

<div id="status">모델 로딩 중… 잠시 기다려 주세요.</div>
<div class="progress-bar"><div class="fill" id="pbar"></div></div>

<div id="gallery">
  <div class="empty-msg">모델 로딩 후 폴더를 선택하고 인덱싱하세요.</div>
</div>

<!-- 이미지 팝업 모달 -->
<div id="imgModal" onclick="closeImageModal()">
  <div id="imgModalInner" onclick="event.stopPropagation()">
    <button id="imgModalClose" onclick="closeImageModal()">✕ 닫기</button>
    <img id="imgModalImg" src=""/>
    <div id="imgModalInfo"></div>
  </div>
</div>

<script>
let selectedFolder = "";
let pollTimer = null;
let dlTimer = null;
let apiReady = false;
let modelReady = false;

// ── pywebview.api 준비 대기 ───────────────────────────────
function waitForApi(callback) {
  if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.init_model === 'function') {
    apiReady = true;
    console.log("[JS] pywebview.api 준비 완료 (메서드 확인됨)");
    callback();
  } else {
    setTimeout(function() { waitForApi(callback); }, 300);
  }
}

// ── 버튼 활성화/비활성화 ──────────────────────────────────
function setButtonsEnabled(enabled) {
  document.getElementById("btnFolder").disabled = !enabled;
  document.getElementById("btnIndex").disabled = !enabled;
  document.getElementById("btnLoadIdx").disabled = !enabled;
  document.getElementById("btnSearch").disabled = !enabled;
  document.getElementById("searchInput").disabled = !enabled;
  document.getElementById("langSelect").disabled = !enabled;
  document.getElementById("btnReset").disabled = !enabled;
}

// ── 다운로드 폴링 ─────────────────────────────────────────
function startDownloadPolling() {
  dlTimer = setInterval(async function() {
    try {
      const d = await pywebview.api.get_download_progress();
      console.log("[JS] download:", d.pct, "%", d.msg);
      setStatus("⬇️ " + d.msg);
      document.getElementById("pbar").style.width = d.pct + "%";

      if (!d.downloading && d.ready) {
        clearInterval(dlTimer);
        dlTimer = null;
        modelReady = true;
        setButtonsEnabled(true);
        document.getElementById("pbar").style.width = "100%";

        // 이전 인덱스 상태 확인
        try {
          const st = await pywebview.api.get_index_state();
          console.log("[JS] index_state:", JSON.stringify(st));
          if (st.has_index && st.count > 0) {
            selectedFolder = st.folder;
            setStatus("✅ 모델 로드 완료! 이전 인덱스 " + st.count + "개 복원됨. 바로 검색 가능합니다.");
            document.getElementById("btnIndex").disabled = false;
            // 복원된 recent 표시
            const p = await pywebview.api.get_progress();
            if (p.recent && p.recent.length > 0) {
              renderRecent(p.recent);
            }
          } else {
            setStatus("✅ 모델 로드 완료! 폴더를 선택하세요.");
            document.getElementById("btnIndex").disabled = true;
          }
        } catch(e) {
          console.error("[JS] index_state 조회 예외:", e);
          setStatus("✅ 모델 로드 완료! 폴더를 선택하세요.");
          document.getElementById("btnIndex").disabled = true;
        }
      } else if (!d.downloading && !d.ready) {
        clearInterval(dlTimer);
        dlTimer = null;
        setStatus("❌ " + d.msg);
        document.getElementById("btnDownload").disabled = false;
      }
    } catch(e) {
      console.error("[JS] download poll 예외:", e);
    }
  }, 500);
}

// ── 앱 시작 ───────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", function() {
  console.log("[JS] DOMContentLoaded");
  setStatus("⏳ pywebview API 준비 대기 중...");
  setButtonsEnabled(false);

  waitForApi(async function() {
    setStatus("⏳ 모델 자동 로딩 중...");
    try {
      console.log("[JS] 자동 init_model 호출");
      const res = await pywebview.api.init_model();
      console.log("[JS] init_model 응답:", JSON.stringify(res));
      if (res.ok) {
        startDownloadPolling();
      } else {
        setStatus("⚠️ " + res.msg);
        document.getElementById("btnDownload").style.display = "inline-block";
        document.getElementById("btnDownload").disabled = false;
      }
    } catch(e) {
      console.error("[JS] 자동 초기화 예외:", e);
      setStatus("❌ 초기화 오류: " + e);
      document.getElementById("btnDownload").style.display = "inline-block";
      document.getElementById("btnDownload").disabled = false;
    }

    // 언어 선택 드롭다운 초기화
    try {
      const langRes = await pywebview.api.get_language_list();
      console.log("[JS] 언어 리스트:", JSON.stringify(langRes));
      if (langRes.ok) {
        const sel = document.getElementById("langSelect");
        sel.innerHTML = "";
        for (const l of langRes.langs) {
          const opt = document.createElement("option");
          opt.value = l.code;
          opt.textContent = l.name;
          if (l.code === langRes.default) {
            opt.selected = true;
          }
          sel.appendChild(opt);
        }
      }
    } catch(e) {
      console.error("[JS] 언어 리스트 로드 예외:", e);
    }
  });
});

// ── 다운로드 버튼 ─────────────────────────────────────────
async function startDownload() {
  console.log("[JS] 다운로드 버튼 클릭");
  try {
    document.getElementById("btnDownload").disabled = true;
    setButtonsEnabled(false);
    const res = await pywebview.api.init_model();
    console.log("[JS] init_model 응답:", JSON.stringify(res));
    if (res.ok) {
      startDownloadPolling();
    } else {
      setStatus("⚠️ " + res.msg);
      document.getElementById("btnDownload").disabled = false;
    }
  } catch(e) {
    console.error("[JS] startDownload 예외:", e);
    setStatus("❌ 다운로드 시작 오류: " + e);
    document.getElementById("btnDownload").disabled = false;
  }
}

// ── 폴더 선택 ─────────────────────────────────────────────
async function selectFolder() {
  console.log("[JS] selectFolder 버튼 클릭");
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    const res = await pywebview.api.select_folder();
    console.log("[JS] select_folder 응답:", JSON.stringify(res));
    if (!res.ok) { setStatus("⚠️ " + res.msg); return; }
    selectedFolder = res.path;
    console.log("[JS] 선택 폴더:", selectedFolder);
    setStatus("📁 선택된 폴더: " + selectedFolder);
    document.getElementById("btnIndex").disabled = false;

    const scan = await pywebview.api.scan_images(selectedFolder);
    console.log("[JS] scan_images 응답:", scan.count, "개");
    setStatus("📁 " + selectedFolder + "  →  이미지 " + scan.count + "개 발견");
  } catch(e) {
    console.error("[JS] selectFolder 예외:", e);
    setStatus("❌ 폴더 선택 오류: " + e);
  }
}

// ── 인덱싱 시작 ───────────────────────────────────────────
async function startIndexing() {
  console.log("[JS] startIndexing 클릭, 폴더:", selectedFolder);
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    if (!selectedFolder) { setStatus("⚠️ 먼저 폴더를 선택하세요."); return; }
    document.getElementById("btnIndex").disabled = true;
    await pywebview.api.start_indexing(selectedFolder);
    setStatus("⚙️ 인덱싱 시작…");
    pollTimer = setInterval(pollProgress, 500);
  } catch(e) {
    console.error("[JS] startIndexing 예외:", e);
    setStatus("❌ 인덱싱 시작 오류: " + e);
  }
}

async function pollProgress() {
  try {
    const p = await pywebview.api.get_progress();
    console.log("[JS] pollProgress:", p.current, "/", p.total);
    setStatus("⚙️ " + p.msg);
    const pct = p.total > 0 ? Math.round(p.current / p.total * 100) : 0;
    document.getElementById("pbar").style.width = pct + "%";

    if (p.recent && p.recent.length > 0) {
      renderRecent(p.recent);
    }

    if (!p.indexing) {
      clearInterval(pollTimer);
      document.getElementById("btnIndex").disabled = false;
      setStatus("✅ " + p.msg);
    }
  } catch(e) {
    console.error("[JS] pollProgress 예외:", e);
  }
}

// ── 저장된 인덱스 로드 ────────────────────────────────────
async function loadIndex() {
  console.log("[JS] loadIndex 클릭");
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    const res = await pywebview.api.load_index();
    console.log("[JS] load_index 응답:", JSON.stringify(res));
    setStatus(res.ok ? "✅ " + res.msg : "⚠️ " + res.msg);
  } catch(e) {
    console.error("[JS] loadIndex 예외:", e);
    setStatus("❌ 인덱스 로드 오류: " + e);
  }
}

// ── 검색 ──────────────────────────────────────────────────
function toggleClearBtn() {
  const v = document.getElementById("searchInput").value;
  document.getElementById("btnClear").style.display = v ? "inline-block" : "none";
}

function clearSearch() {
  const inp = document.getElementById("searchInput");
  inp.value = "";
  toggleClearBtn();
  inp.focus();
}

async function showAll() {
  try {
    setStatus("📋 전체 목록 조회 중...");
    const res = await pywebview.api.list_all();
    console.log("[JS] list_all 응답:", res.ok, res.results ? res.results.length : 0);
    if (!res.ok) { setStatus("⚠️ " + res.msg); return; }
    renderAll(res.results);
    setStatus("📋 전체 목록 " + res.results.length + "개");
  } catch(e) {
    console.error("[JS] showAll 예외:", e);
    setStatus("❌ 전체 목록 오류: " + e);
  }
}

function renderAll(items) {
  const g = document.getElementById("gallery");
  if (!items || items.length === 0) {
    g.innerHTML = '<div class="empty-msg">인덱스에 항목이 없습니다.</div>';
    return;
  }
  let html = '<div style="grid-column:1/-1;color:#7eb8ff;font-size:13px;padding:4px 0;">📋 전체 목록 (' + items.length + '개)</div>';
  for (const r of items) {
    const imgSrc = r.thumb_b64 || "";
    const safeName = (r.name || "").replace(/'/g, "\\'");
    html += `
      <div class="card">
        <img src="${imgSrc}"
             onclick="showImageModal(this.src, '${safeName}', '')"
             onerror="this.style.display='none'"/>
        <div class="info">📄 ${r.name}</div>
      </div>`;
  }
  g.innerHTML = html;
}

async function doSearch() {
  const q = document.getElementById("searchInput").value.trim();
  console.log("[JS] doSearch:", q);
  if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
  if (!q) { await showAll(); return; }
  try {
    setStatus("🔍 검색 중: " + q);
    const res = await pywebview.api.search(q, 20);
    console.log("[JS] search 응답:", res.ok, res.results ? res.results.length : 0, "개");
    if (!res.ok) { setStatus("❌ " + res.msg); return; }
    
    // 결과가 없으면 갤러리 비우기
    if (!res.results || res.results.length === 0) {
      renderResults([]);
      setStatus("🔍 \"" + q + "\" → 검색 결과 없음");
      return;
    }
    
    renderResults(res.results);
    const transInfo = res.translated && res.translated !== q
      ? " (번역: " + res.translated + ")"
      : "";
    setStatus("🔍 \"" + q + "\"" + transInfo + " → " + res.results.length + "개 결과");
  } catch(e) {
    console.error("[JS] doSearch 예외:", e);
    setStatus("❌ 검색 오류: " + e);
  }
}

function renderResults(results) {
  const g = document.getElementById("gallery");
  if (!results || results.length === 0) {
    g.innerHTML = '<div class="empty-msg">검색 결과가 없습니다.</div>';
    return;
  }
  let html = "";
  for (const r of results) {
    // cosine(0~1)는 % 표기, sparse 원점수(1 초과)는 소수점 3자리 표기
    const scoreTxt = (typeof r.score === "number" && r.score <= 1.0)
      ? (r.score * 100).toFixed(1) + "%"
      : (r.score || 0).toFixed(3);
    const imgSrc = r.thumb_b64 || "";
    const safeName = (r.name || "").replace(/'/g, "\\'");
    html += `
      <div class="card">
        <img src="${imgSrc}"
             onclick="showImageModal(this.src, '${safeName}', '${scoreTxt}')"
             onerror="this.style.display='none'"/>
        <div class="info">
          <span class="score">${scoreTxt}</span> · ${r.name}
        </div>
      </div>`;
  }
  g.innerHTML = html;
}

function renderRecent(items) {
  const g = document.getElementById("gallery");
  if (!items || items.length === 0) return;
  let html = '<div style="grid-column:1/-1;color:#7eb8ff;font-size:13px;padding:4px 0;">📌 인덱싱 완료된 이미지 (최신순)</div>';
  for (const r of items) {
    const imgSrc = r.thumb_b64 || "";
    const safeName = (r.name || "").replace(/'/g, "\\'");
    html += `
      <div class="card">
        <img src="${imgSrc}"
             onclick="showImageModal(this.src, '${safeName}', '')"
             onerror="this.style.display='none'"/>
        <div class="info">✅ ${r.name}</div>
      </div>`;
  }
  g.innerHTML = html;
}

async function changeLanguage() {
  const sel = document.getElementById("langSelect");
  const langCode = sel.value;
  console.log("[JS] 언어 변경:", langCode);
  try {
    setStatus("🌐 번역 모델 로딩 중: " + sel.options[sel.selectedIndex].text);
    const res = await pywebview.api.load_translator(langCode);
    console.log("[JS] load_translator 응답:", JSON.stringify(res));
    if (res.ok) {
      setStatus("✅ " + res.msg);
    } else {
      setStatus("⚠️ " + res.msg);
    }
  } catch(e) {
    console.error("[JS] changeLanguage 예외:", e);
    setStatus("❌ 언어 변경 오류: " + e);
  }
}

// ── 이미지 팝업 모달 ──────────────────────────────────────
function showImageModal(src, name, score) {
  if (!src) return;
  const modal = document.getElementById("imgModal");
  const modalImg = document.getElementById("imgModalImg");
  const modalInfo = document.getElementById("imgModalInfo");
  modalImg.src = src;
  modalInfo.textContent = name + (score ? " · " + score : "");
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeImageModal() {
  const modal = document.getElementById("imgModal");
  modal.style.display = "none";
  document.getElementById("imgModalImg").src = "";
  document.body.style.overflow = "";
}

document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") {
    closeImageModal();
  }
});

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

async function resetIndex() {
    if (!confirm("인덱스를 완전히 초기화할까요?")) return;
    try {
        const res = await pywebview.api.reset_index();
        setStatus(res.ok ? "✅ " + res.msg : "⚠️ " + res.msg);
        if (res.ok) {
            document.getElementById("gallery").innerHTML =
                '<div class="empty-msg">인덱스가 초기화되었습니다. 폴더를 선택하고 인덱싱하세요.</div>';
        }
    } catch(e) {
        console.error("[JS] resetIndex 예외:", e);
        setStatus("❌ 초기화 오류: " + e);
    }
}
</script>
</body>
</html>
"""

# ============================================================
#  로컬 파일 서빙
# ============================================================
def local_file_handler(path: str):
    import urllib.parse
    real = path.replace("local-file://", "")
    real = urllib.parse.unquote(real)
    if os.name == "nt":
        real = real.lstrip("/")
    else:
        if not real.startswith("/"):
            real = "/" + real
    if os.path.isfile(real):
        return real
    return None

# ============================================================
#  메인
# ============================================================
def main():
    logger.info("[메인] PyWebView 창 생성 시작")
    api = Api()
    window = webview.create_window(
        "V-SPLADE Vision Search",
        html=HTML_PAGE,  # 이전 완성본
        js_api=api,
        width=1200, height=800, min_size=(900, 600),
    )
    logger.info("[메인] PyWebView 시작")
    webview.start(debug=True)

if __name__ == "__main__":
    main()