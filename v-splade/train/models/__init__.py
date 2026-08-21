# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V_SPLADE modular components.

A V_SPLADE retriever is composed of:
    Encoder + Pooling + SparseHead + (BOW | Li-LSR) QueryEncoder + Losses
"""

from pathlib import Path

import torch

from models.model import UnifiedRetriever, RetrievalOutput, compute_logits
from models.encoder import EncoderType, build_encoder
from models.pooling import PoolingType, Pooling
from models.head import HeadType, build_head, SparseHead
from models.query_encoder import (
    QueryEncoderType,
    build_query_encoder,
    BOWQueryEncoder,
    InferenceFreeQueryEncoder,
)
from models.losses import FLOPSLoss, NCELoss, CaptionPushUpLoss


DEFAULT_POOLING = {
    "vbert": "max",
}


def build_model(
    path: str = None,
    mode: str = "inference_only",
    *,
    encoder_type: str = "vbert",
    head_type: str = "sparse",
    query_encoder_type: str = "li_lsr",
    pooling_type: str = None,
    query_lsr_lora_r: int = 0,
    query_lsr_activation: str = "softplus",
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> UnifiedRetriever:
    """Factory: build a V-SPLADE retriever in one of two modes.

    ``mode='inference_only'`` (default):
        ``path`` is a V-SPLADE HF export directory containing
        ``model.safetensors`` + ``config.json``. The retriever is constructed
        as an empty shell and every weight (backbone + SPLADE head + Li-LSR
        query head) is dispatched from the export in a single pass. No base
        model download, no LoRA wrapping.

    ``mode='from_scratch'``:
        ``path`` is the base BiModernVBert backbone directory (e.g. the
        canonical ``ModernVBERT/modernvbert`` checkpoint). The retriever is
        built for training — encoder/LM-head LoRA, fresh query head,
        loss/regularizer hooks. Extra ``**kwargs`` are forwarded to
        :class:`UnifiedRetriever`.
    """
    if mode == "inference_only":
        if path is None:
            raise ValueError("inference_only mode requires path= to the HF export dir")
        model = UnifiedRetriever.from_hf_export(
            path,
            query_lsr_activation=query_lsr_activation,
            dtype=dtype,
        )
        load_hf_export(model, path, dtype=dtype)
        return model

    if mode == "from_scratch":
        if pooling_type is None:
            pooling_type = DEFAULT_POOLING.get(encoder_type, "max")
        if path is not None:
            kwargs.setdefault("model_name", path)
        return UnifiedRetriever(
            encoder_type=encoder_type,
            pooling_type=pooling_type,
            head_type=head_type,
            query_encoder_type=query_encoder_type,
            query_lsr_lora_r=query_lsr_lora_r,
            query_lsr_activation=query_lsr_activation,
            **kwargs,
        )

    raise ValueError(f"Unknown mode: {mode!r}. Choose 'inference_only' or 'from_scratch'.")


def _resolve_export_file(hf_dir: str, filename: str) -> str:
    """Resolve a file from a V-SPLADE HF export given as a local dir or Hub id.

    Returns a local filesystem path: the file under ``hf_dir`` if it exists,
    otherwise the result of downloading ``filename`` from the Hub repo
    ``hf_dir`` (cached by huggingface_hub).
    """
    local = Path(hf_dir) / filename
    if local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=str(hf_dir), filename=filename)


def load_hf_export(model: UnifiedRetriever, hf_dir: str,
                   dtype: torch.dtype = torch.bfloat16) -> None:
    from safetensors.torch import load_file

    full_sd = load_file(_resolve_export_file(hf_dir, "model.safetensors"))

    dispatch = [
        (model.encoder,       "encoder."),
        (model.query_encoder, "query_encoder."),
    ]

    consumed = set()
    for module, prefix in dispatch:
        # 원본에서 pop하며 이동 → 체크포인트 사본이 중복 상주하지 않음(피크 메모리 절반)
        slice_sd = {}
        for k in list(full_sd.keys()):
            if k.startswith(prefix):
                v = full_sd.pop(k)
                slice_sd[k[len(prefix):]] = v if v.dtype == dtype else v.to(dtype)
                del v

        all_slice_keys = list(slice_sd.keys())

        # ── encoder: tok_embeddings = main + additional concat ──
        if prefix == "encoder.":
            _TOK_KEY = "encoder.model.text_model.embeddings.tok_embeddings.weight"
            _ADD_KEY = "encoder.model.text_model.embeddings.tok_embeddings.additional_embedding.weight"

            if _TOK_KEY in slice_sd:
                main_w = slice_sd[_TOK_KEY]
                add_w = slice_sd.get(_ADD_KEY)
                full_w = torch.cat([main_w, add_w], dim=0) if add_w is not None else main_w

                try:
                    target_size = module.encoder.model.text_model.embeddings.tok_embeddings.weight.size(0)
                except AttributeError:
                    target_size = full_w.size(0)

                if full_w.size(0) < target_size:
                    p = torch.zeros(target_size, full_w.size(1), dtype=full_w.dtype, device=full_w.device)
                    p[:full_w.size(0)] = full_w
                    full_w = p
                elif full_w.size(0) > target_size:
                    full_w = full_w[:target_size]
                slice_sd[_TOK_KEY] = full_w
            slice_sd.pop(_ADD_KEY, None)

        incompat = module.load_state_dict(slice_sd, strict=False)

        # ★ 핵심: 이름 불일치로 누락된 커넥터 가중치를 shape 매핑으로 주입
        if getattr(incompat, "missing_keys", None):
            sd = module.state_dict()
            remapped = []
            for miss_key in list(incompat.missing_keys):
                if "connector" not in miss_key or not miss_key.endswith(".weight"):
                    continue
                t_shape = tuple(sd[miss_key].shape)
                for ck, cv in list(slice_sd.items()):
                    if tuple(cv.shape) == t_shape:
                        sd[miss_key] = cv.to(sd[miss_key].dtype)
                        remapped.append((ck, miss_key))
                        del slice_sd[ck]
                        break
            if remapped:
                module.load_state_dict(sd, strict=False)
                for ck, mk in remapped:
                    print(f"[load_hf_export] remapped: {ck} -> {mk}")

        consumed.update(prefix + k for k in all_slice_keys)

    leftover = set(full_sd) - consumed
    if leftover:
        raise RuntimeError(
            f"{len(leftover)} tensor(s) in {hf_dir}/model.safetensors were not "
            f"dispatched to any sub-module. First few: {sorted(leftover)[:3]}"
        )
    del full_sd, slice_sd