import os, sys, torch
from pathlib import Path
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "v-splade" / "examples"))
MODEL_DIR = os.path.join(os.environ["LOCALAPPDATA"], "VSpladeSearch", "vsplade-quality")

from vsplade_inference import VSPLADEInference
model = VSPLADEInference.from_pretrained(MODEL_DIR, device="cuda", dtype=torch.bfloat16)

# ── 1) 체크포인트 vs 실제 모델 가중치 전수 비교 ──
from safetensors.torch import load_file
ckpt = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
mod = model.model.state_dict()

missing, bad = [], []
for k, v in ckpt.items():
    if k not in mod:
        missing.append(k); continue
    dv = mod[k].detach().cpu().float()
    cv = v.cpu().float()
    if dv.shape != cv.shape:
        bad.append((k, f"shape {tuple(cv.shape)} vs {tuple(dv.shape)}"))
    elif (dv - cv).abs().max().item() > 1e-4:
        bad.append((k, f"maxdiff={(dv - cv).abs().max().item():.4f}"))

print("=== NOT IN MODEL (안 들어감) ===")
for k in missing: print("  ", k)
print("=== VALUE MISMATCH ===")
for k, r in bad: print("  ", k, r)
print(f"ckpt={len(ckpt)} missing={len(missing)} mismatch={len(bad)}")

# ── 2) 현재 경로의 이미지 토 ──
img = Image.open(r"C:\Users\HP\Desktop\sample\test.jpg").convert("RGB")
vec = model.encode_image(img)
print("current path top tokens:", model.decode_topk(vec, k=10))

# ── 3) 공식 릴리즈 코드(trust_remote_code) 경로 테스트 ──
try:
    from transformers import AutoModel
    am = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
    am = am.eval().to("cuda", torch.bfloat16)
    print("trust_remote_code load OK:", type(am).__name__)
    print("methods:", [m for m in dir(am) if "encode" in m or "forward" in m])
except Exception as e:
    print("trust_remote_code path FAILED:", type(e).__name__, e)