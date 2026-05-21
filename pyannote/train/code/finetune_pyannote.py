import os, sys, json, math, logging, contextlib, tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from itertools import permutations as _perms
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.optim import AdamW

try:
    if hasattr(torchaudio, "set_audio_backend"):
        for _backend in ("sox_io", "soundfile"):
            try:
                torchaudio.set_audio_backend(_backend)
                break
            except Exception:
                continue
except Exception:
    pass

def _load_audio(path: str):
    try:
        return torchaudio.load(path)
    except Exception as e:
        message = str(e)
        if "torchcodec" in message or "TorchCodec" in message:
            for _backend in ("sox_io", "soundfile"):
                try:
                    if hasattr(torchaudio, "set_audio_backend"):
                        torchaudio.set_audio_backend(_backend)
                    return torchaudio.load(path)
                except Exception:
                    continue
            try:
                import soundfile as sf
                data, sr = sf.read(path, dtype="float32")
                data = np.asarray(data)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                else:
                    data = data.T
                return torch.from_numpy(data), sr
            except Exception:
                pass
        raise

from scipy.optimize import linear_sum_assignment

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyanote.ft_config import (
    FinetuneConfig, PhaseConfig, build_phase_configs,
    TEST_MODE, SOURCE_SELF, SOURCE_TIER_A,
    SELF_LABELED_DIR, SELF_LABELED_META,
    DATA_MIX_DIR,  DATA_MIX_META,
    VAL_DATA_DIR, TEST_DATA_DIR, VAL_LABELED_DIR, TEST_LABELED_DIR,
)
from ft_dataloader import (
    Sample,
    scan_self_labeled, scan_all_sources, scan_fixed_split_dir,
    train_val_split, parse_rttm,
    build_train_loader, build_val_loader,
    AudioAugmenter, SR, FRAME_SHIFT_SEC,
)

def _get_device_type(device: torch.device) -> str:
    if device.type == "cuda": return "cuda"
    if device.type == "mps":  return "mps"
    return "cpu"

def _resolve_amp_dtype(device_type: str, amp_dtype: Optional[str]):
    if device_type != "cuda":
        return None
    name = str(amp_dtype or "float16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16

def _use_grad_scaler(enabled: bool, device_type: str, amp_dtype: Optional[str]) -> bool:
    return bool(enabled and device_type == "cuda" and _resolve_amp_dtype(device_type, amp_dtype) == torch.float16)

try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast_fn
    def _make_scaler(enabled, device_type, amp_dtype=None):
        return _GradScaler(device_type, enabled=_use_grad_scaler(enabled, device_type, amp_dtype))
    def _autocast(enabled, device_type="cpu", amp_dtype=None):
        kwargs = {"enabled": enabled}
        dtype = _resolve_amp_dtype(device_type, amp_dtype)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return _autocast_fn(device_type, **kwargs)
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast_fn
    def _make_scaler(enabled, device_type, amp_dtype=None):
        return _GradScaler(enabled=_use_grad_scaler(enabled, device_type, amp_dtype))
    def _autocast(enabled, device_type="cpu", amp_dtype=None):
        if device_type != "cuda":
            return contextlib.nullcontext()
        dtype = _resolve_amp_dtype(device_type, amp_dtype)
        return _autocast_fn(enabled=enabled, dtype=dtype)

_log_dir = _REPO_ROOT / "results" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "finetune_pyannote.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ft")

def load_ecapa_model():

    try:
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        _ecapa_savedir = str(Path(tempfile.gettempdir()) / "sb_ecapa")
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_ecapa_savedir,
        )
        ecapa.eval()
        log.info("  [enroll] ECAPA-TDNN loaded ")
        return ecapa
    except Exception as e:
        log.warning(f"  [enroll] ECAPA not available ({e}). EnrollConf metric disabled.")
        return None

class PyannoteSegWrapper(nn.Module):
    def __init__(self, pipeline, max_speakers: int = 3):
        super().__init__()
        self.pipeline     = pipeline
        self.max_speakers = max_speakers

        self._seg_model              = None
        self._unfreeze_layers        = None
        self._head_param_names: set  = set()
        self._powerset               = None
        self._num_powerset_classes   = 0
        self._max_spk_per_chunk      = max_speakers
        self._max_spk_per_frame      = 2
        self._batched_forward_warned = False

        self._probe()

    def _probe(self):

        seg_model = getattr(self.pipeline, "segmentation_model", None)

        if hasattr(seg_model, "model") and not isinstance(seg_model, nn.Module):
            seg_model = seg_model.model

        if seg_model is None or not isinstance(seg_model, nn.Module):

            _seg = getattr(self.pipeline, "_segmentation", None)
            if hasattr(_seg, "model"):
                seg_model = _seg.model
            elif hasattr(_seg, "model_"):
                seg_model = _seg.model_

        if seg_model is None or not isinstance(seg_model, nn.Module):
            raise RuntimeError(
                "Không tìm thấy segmentation_model trong pipeline. "
                "Đảm bảo model là pyannote/speaker-diarization-community-1 hoặc tương đương."
            )
        self._seg_model = seg_model
        log.info(f"  [probe] segmentation_model: {type(seg_model).__name__}")

        specs = getattr(seg_model, "specifications", None)
        if specs is None:
            raise RuntimeError(
                "model.specifications không tồn tại. "
                "Đây có phải là pyannote segmentation model không?"
            )
        log.info(f"Specifications dir: {dir(specs)}")
        self._max_spk_per_chunk = int(getattr(specs, "max_speakers_per_chunk", getattr(specs, "num_classes", len(getattr(specs, "classes", [])))))
        self._max_spk_per_frame = int(getattr(specs, "max_speakers_per_frame", getattr(specs, "max_speakers_per_time", 2)))
        self._num_powerset_classes = int(getattr(specs, "num_powerset_classes", getattr(specs, "num_classes", 0)))

        try:
            from pyannote.audio.utils.powerset import Powerset
            self._powerset = Powerset(
                self._max_spk_per_chunk,
                self._max_spk_per_frame,
            )
            log.info(
                f"  [probe] Powerset: max_spk/chunk={self._max_spk_per_chunk}, "
                f"max_spk/frame={self._max_spk_per_frame}, "
                f"num_powerset_classes={self._num_powerset_classes}"
            )
        except ImportError:
            raise RuntimeError("pyannote.audio.utils.powerset không tìm thấy — pip install pyannote.audio")

        for attr_name in ("linear", "classifier", "projection"):
            head = getattr(seg_model, attr_name, None)
            if head is not None and isinstance(head, nn.Linear):
                for suffix in ("weight", "bias"):
                    self._head_param_names.add(f"_seg_model.{attr_name}.{suffix}")
                log.info(
                    f"  [probe] Head: {attr_name} "
                    f"Linear({head.in_features}, {head.out_features})"
                )
                break
        if not self._head_param_names:
            log.warning("  [probe] HEAD không tìm thấy — chỉ backbone params được freeze/unfreeze.")

        _ssl_attrs = ("wav2vec2", "wavlm", "hubert", "feature_extractor", "encoder", "backbone")
        for attr in _ssl_attrs:
            backbone = getattr(seg_model, attr, None)
            if backbone is None or not isinstance(backbone, nn.Module):
                continue

            enc = getattr(backbone, "encoder", backbone)
            for la in ("layers", "blocks", "transformer_layers"):
                layers = getattr(enc, la, None)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    self._unfreeze_layers = layers
                    log.info(
                        f"  [probe] SSL backbone: {len(layers)} transformer layers "
                        f"via seg_model.{attr}.encoder.{la}"
                    )
                    return

            for la in ("layers", "blocks"):
                layers = getattr(backbone, la, None)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    self._unfreeze_layers = layers
                    log.info(
                        f"  [probe] SSL backbone: {len(layers)} layers "
                        f"via seg_model.{attr}.{la}"
                    )
                    return

        sincnet = getattr(seg_model, "sincnet", None)
        lstm    = getattr(seg_model, "lstm",    None)
        if lstm is not None:
            layer_groups: List[nn.Module] = []
            if sincnet is not None:
                layer_groups.append(sincnet)

            num_lstm_layers = getattr(lstm, "num_layers", 1)
            if num_lstm_layers > 1:
                class _LSTMLayerProxy(nn.Module):

                    def __init__(self, lstm_module: nn.Module, layer_idx: int):
                        super().__init__()
                        self._lstm = lstm_module
                        self._layer_idx = layer_idx

                        suffixes = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
                        for sfx in suffixes:
                            for direction in ([""] if not getattr(lstm_module, "bidirectional", False)
                                              else ["", "_reverse"]):
                                pname = f"{sfx}_l{layer_idx}{direction}"
                                param = getattr(lstm_module, pname, None)
                                if param is not None:
                                    self.register_parameter(pname.replace(".", "_"), param)

                    def parameters(self, recurse=True):

                        suffixes = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
                        seen = set()
                        for sfx in suffixes:
                            for direction in ([""] if not getattr(self._lstm, "bidirectional", False)
                                              else ["", "_reverse"]):
                                pname = f"{sfx}_l{self._layer_idx}{direction}"
                                param = getattr(self._lstm, pname, None)
                                if param is not None and id(param) not in seen:
                                    seen.add(id(param))
                                    yield param

                for li in range(num_lstm_layers):
                    layer_groups.append(_LSTMLayerProxy(lstm, li))
            else:
                layer_groups.append(lstm)

            self._unfreeze_layers = nn.ModuleList(layer_groups) if layer_groups else None
            log.info(
                f"  [probe] SincNet+LSTM model: "
                f"{len(layer_groups)} unfreezable layer groups "
                f"(sincnet={'yes' if sincnet else 'no'}, lstm_layers={num_lstm_layers})"
            )
            return

        log.warning(
            "  [probe] Không tìm thấy backbone layers cho gradual unfreeze. "
            "Chỉ head params sẽ được train ở phase đầu."
        )

    def is_head_param(self, name: str) -> bool:
        return name in self._head_param_names

    def unfreeze_layers(self):
        return self._unfreeze_layers

    def powerset(self):
        return self._powerset

    def _extract_logits(self, raw_out) -> torch.Tensor:

        if isinstance(raw_out, tuple):
            logits = raw_out[0]
        elif isinstance(raw_out, dict):
            logits = raw_out.get("logits", next(iter(raw_out.values())))
        else:
            logits = raw_out

        if logits.ndim == 3:
            return logits
        if logits.ndim == 2:
            return logits.unsqueeze(0)
        return logits.view(1, -1, self._num_powerset_classes)

    def _forward_one(self, wav: torch.Tensor) -> torch.Tensor:
        wav_3d = wav.view(1, 1, -1)
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            raw_out = self._seg_model(wav_3d)
        return self._extract_logits(raw_out).squeeze(0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        try:
            with torch.set_grad_enabled(torch.is_grad_enabled()):
                raw_out = self._seg_model(waveform.contiguous())
            return self._extract_logits(raw_out)
        except Exception as e:
            B = waveform.shape[0]
            if B == 1:
                raise RuntimeError(f"Model forward failed: {e}")
            if not self._batched_forward_warned:
                log.warning(
                    f"  [forward] Batched forward failed ({e}). "
                    f"Falling back to per-sample forward."
                )
                self._batched_forward_warned = True
            outs = [self._forward_one(waveform[b]) for b in range(B)]
            max_T = max(x.shape[0] for x in outs)
            logits = waveform.new_zeros(B, max_T, self._num_powerset_classes)
            for b, x in enumerate(outs):
                logits[b, :x.shape[0]] = x
            return logits

_PERM_CACHE: Dict[int, List] = {}
def get_perms(n: int) -> List:
    if n not in _PERM_CACHE:
        _PERM_CACHE[n] = list(_perms(range(n)))
    return _PERM_CACHE[n]

def decode_probs(
    logits: torch.Tensor,
    powerset: object,
) -> torch.Tensor:

    probs = torch.softmax(logits, dim=-1)
    if hasattr(powerset, "mapping") and probs.device != powerset.mapping.device:
        probs = probs.to(powerset.mapping.device)
    out_probs = powerset.to_multilabel(probs)
    if hasattr(powerset, "mapping") and out_probs.device != logits.device:
        out_probs = out_probs.to(logits.device)
    return out_probs

def decode_probs_argmax(
    logits: torch.Tensor,
    powerset: object,
) -> torch.Tensor:

    idx = logits.argmax(dim=-1)
    onehot = F.one_hot(idx, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    if hasattr(powerset, "mapping") and onehot.device != powerset.mapping.device:
        onehot = onehot.to(powerset.mapping.device)
    probs = powerset.to_multilabel(onehot)
    if hasattr(powerset, "mapping") and probs.device != logits.device:
        probs = probs.to(logits.device)
    return probs

def _sync_labels_to_output_grid(
    labels: torch.Tensor,
    n_frames: torch.Tensor,
    target_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if labels.shape[1] != target_frames:
        lbl_r = F.interpolate(
            labels.permute(0, 2, 1).float(), size=target_frames, mode="nearest"
        ).permute(0, 2, 1)
        nf_r = (n_frames.float() * target_frames / max(labels.shape[1], 1)).long().clamp(max=target_frames)
        return lbl_r, nf_r
    return labels.float(), n_frames

def _mask_like(x, mask: torch.Tensor):
    if x is None or not torch.is_tensor(x):
        return x
    if x.device != mask.device:
        return x[mask.detach().cpu()]
    return x[mask]

def _check_task_compatible_batch(
    labels: torch.Tensor,
    n_frames: torch.Tensor,
    max_chunk_speakers: int,
    max_frame_speakers: int,
    policy: str = "raise",
) -> Dict[str, int]:

    skipped_chunk = skipped_frame = skipped_both = skipped_empty = 0
    for b in range(labels.shape[0]):
        L = int(n_frames[b].item())
        if L <= 0:
            skipped_empty += 1
            continue
        lbl_b = labels[b, :L].float()
        active_speakers = int((lbl_b.sum(0) > 0).sum().item())
        max_overlap = int(lbl_b.sum(-1).max().item()) if L > 0 else 0
        chunk_bad = active_speakers > max_chunk_speakers
        frame_bad = max_overlap > max_frame_speakers

        if chunk_bad:
            skipped_chunk += 1
        if frame_bad:
            skipped_frame += 1
        if chunk_bad and frame_bad:
            skipped_both += 1

    n_invalid = skipped_chunk + skipped_frame - skipped_both
    stats = {
        "checked": int(labels.shape[0]),
        "invalid": int(n_invalid),
        "invalid_chunk_speakers": int(skipped_chunk),
        "invalid_frame_overlap": int(skipped_frame),
        "invalid_both": int(skipped_both),
        "empty": int(skipped_empty),
    }

    if stats["invalid"] > 0:
        message = (
            "Batch violates powerset task constraints: "
            f"invalid={stats['invalid']} | "
            f">chunk_cap={stats['invalid_chunk_speakers']} | "
            f">frame_cap={stats['invalid_frame_overlap']} | "
            f"caps=(chunk<={max_chunk_speakers}, frame<={max_frame_speakers}). "
            "This code treats `max_chunk_speakers` as an upper bound, not an exact speaker count."
        )
        if str(policy).lower() == "warn":
            log.warning("  [task_check] " + message)
        elif str(policy).lower() != "ignore":
            raise RuntimeError(message)

    return stats

def powerset_pit_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_frames: torch.Tensor,
    phase: PhaseConfig,
    powerset: object,
    model_max_spk: int,
    model_max_frame_spk: int,
) -> torch.Tensor:

    B = logits.shape[0]
    S_model = model_max_spk
    losses = []

    for b in range(B):
        L = int(n_frames[b].item())
        if L <= 0:
            losses.append(logits[b].sum() * 0.0)
            continue

        log_b = logits[b, :L]
        lbl_full = labels[b, :L].to(logits.device).float()

        active_cols = torch.where(lbl_full.sum(0) > 0)[0]
        if active_cols.numel() > S_model:
            losses.append(logits[b].sum() * 0.0)
            continue

        if active_cols.numel() > 0:
            lbl_b = lbl_full[:, active_cols]
        else:
            lbl_b = lbl_full[:, :0]

        K = lbl_b.shape[-1]
        if K < S_model:
            pad = torch.zeros(L, S_model - K, device=logits.device, dtype=lbl_full.dtype)
            lbl_b = torch.cat([lbl_b, pad], dim=-1)

        n_act = lbl_b.sum(-1)
        valid_frame_mask = n_act <= float(model_max_frame_spk)
        if not torch.any(valid_frame_mask):
            losses.append(logits[b].sum() * 0.0)
            continue

        w = torch.ones(L, device=logits.device, dtype=log_b.dtype)
        w[n_act == 0] = phase.silence_loss_weight
        w[n_act == 2] = phase.overlap_loss_weight
        w[~valid_frame_mask] = 0.0

        ign_frm = int(phase.boundary_ignore_sec / FRAME_SHIFT_SEC)
        if ign_frm > 0:
            w[:min(ign_frm, L)] *= 0.25
            w[max(0, L - ign_frm):] *= 0.25

        best_loss = None
        for perm in get_perms(S_model):
            lbl_perm = lbl_b[:, list(perm)]
            with torch.no_grad():
                ps_onehot = powerset.to_powerset(lbl_perm)
            ps_idx = ps_onehot.argmax(dim=-1)
            ce = F.cross_entropy(log_b, ps_idx, reduction="none")
            perm_loss = (ce * w).sum() / w.sum().clamp(min=1e-8)
            if best_loss is None or perm_loss.item() < best_loss.item():
                best_loss = perm_loss

        losses.append(best_loss if best_loss is not None else logits[b].sum() * 0.0)

    return torch.stack(losses).mean() if losses else logits.sum() * 0.0

def pit_der_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_frames: torch.Tensor,
    powerset: object,
    load_speakers: Optional[torch.Tensor] = None,
) -> Dict[str, float]:

    probs = decode_probs_argmax(logits, powerset)
    S_ps = probs.shape[-1]

    miss = fa = conf = total = 0.0

    for b in range(logits.shape[0]):
        L = int(n_frames[b].item())
        if L <= 0:
            continue

        p = probs[b, :L, :S_ps].float()
        g_full = labels[b, :L].float()
        active_cols = torch.where(g_full.sum(0) > 0)[0]
        if active_cols.numel() > S_ps:
            active_cols = active_cols[:S_ps]
        g = g_full[:, active_cols] if active_cols.numel() > 0 else g_full[:, :0]
        if g.shape[-1] < S_ps:
            pad = torch.zeros(L, S_ps - g.shape[-1], dtype=g.dtype)
            g = torch.cat([g, pad], dim=-1)

        cost_matrix = torch.zeros((S_ps, S_ps))
        for i in range(S_ps):
            for j in range(S_ps):
                cost_matrix[i, j] = (g[:, i] - p[:, j]).abs().sum().item()
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        inv_perm = np.zeros(S_ps, dtype=int)
        for i, j in zip(row_ind, col_ind):
            inv_perm[j] = i
        g_perm = g[:, inv_perm]

        pred_sum = p.sum(-1)
        gold_sum = g_perm.sum(-1)

        total += (gold_sum > 0).float().sum().item()
        miss  += F.relu(gold_sum - pred_sum).sum().item()
        fa    += F.relu(pred_sum - gold_sum).sum().item()
        conf  += F.relu(torch.min(pred_sum, gold_sum) - (p * g_perm).sum(-1)).sum().item()

    return {
        "miss_frames":  miss,
        "fa_frames":    fa,
        "conf_frames":  conf,
        "total_frames": max(total, 1e-8),
    }

def apply_freeze(model: PyannoteSegWrapper, strategy: str, unfreeze_top_n: int):

    for p in model.parameters():
        p.requires_grad = False

    head_unlocked = []
    for name, p in model.named_parameters():
        if model.is_head_param(name):
            p.requires_grad = True
            head_unlocked.append(name)
    log.info(f"  [freeze] Head params unlocked: {head_unlocked}")

    if strategy == "none":

        pass

    elif strategy == "all":

        for p in model.parameters():
            p.requires_grad = True
        log.info("  [freeze] strategy=all → toàn bộ params unlocked")

    elif strategy == "top_N":
        layers = model.unfreeze_layers()
        if layers is not None:
            n = len(layers)
            for i, layer in enumerate(layers):
                if i >= max(0, n - unfreeze_top_n):
                    for p in layer.parameters():
                        p.requires_grad = True
                    log.info(f"  [freeze] Unlocked layer {i}/{n-1}")
        else:
            log.warning(
                "  [freeze] _unfreeze_layers is None! top_N unfreeze skipped. "
                "Chỉ head params được train."
            )

    else:

        raise ValueError(
            f"  [freeze] Unknown freeze_strategy='{strategy}'. "
            "Các giá trị hợp lệ: 'none' (chỉ head), 'all' (toàn bộ), 'top_N' (top-N layers + head)."
        )

    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = sum(p.numel() for p in model.parameters())
    log.info(f"  [freeze] {strategy} | trainable: {t:,}/{n:,} ({100*t/max(n,1):.1f}%)")

def build_optimizer(model: PyannoteSegWrapper, phase: PhaseConfig) -> AdamW:
    head_names  = [(nm, p) for nm, p in model.named_parameters()
                   if p.requires_grad and model.is_head_param(nm)]
    other_names = [(nm, p) for nm, p in model.named_parameters()
                   if p.requires_grad and not model.is_head_param(nm)]

    head_p  = [p for _, p in head_names]
    other_p = [p for _, p in other_names]

    log.info(f"  [optimizer] HEAD params ({len(head_p)} tensors):")
    for nm, _ in head_names:
        log.info(f"    {nm}")
    log.info(f"  [optimizer] OTHER trainable params ({len(other_p)} tensors):")
    for nm, _ in other_names[:8]:
        log.info(f"    {nm}")
    if len(other_names) > 8:
        log.info(f"    ... ({len(other_names)-8} more)")

    groups = []
    if head_p:
        groups.append({"params": head_p, "lr": phase.lr})
    if other_p:
        groups.append({"params": other_p, "lr": phase.lr * 0.2})
    if not groups:
        raise RuntimeError("No trainable parameters found. Check freeze strategy.")

    log.info(
        f"  [optimizer] groups={len(groups)} | "
        + " | ".join(f"lr={g['lr']:.2e} ({len(g['params'])} tensors)" for g in groups)
    )
    return AdamW(groups, weight_decay=phase.weight_decay, betas=(0.9, 0.999), eps=1e-8)

class WarmupCosine:
    def __init__(self, opt, warmup: int, total: int, min_r: float = 0.05):
        self.opt, self.warmup, self.total, self.min_r, self._step = opt, warmup, total, min_r, 0
        self._base = [g["lr"] for g in opt.param_groups]

    def step(self):
        self._step += 1; s = self._step
        for i, g in enumerate(self.opt.param_groups):
            base = self._base[i]
            if s < self.warmup:
                g["lr"] = base * s / max(self.warmup, 1)
            else:
                p = (s - self.warmup) / max(self.total - self.warmup, 1)
                g["lr"] = base * (self.min_r + (1 - self.min_r) * 0.5 * (1 + math.cos(math.pi * p)))

    def get_lr(self):
        return [g["lr"] for g in self.opt.param_groups]

class ValRegistry:
    def __init__(self, val_files: List[Sample], cfg: FinetuneConfig):
        from ft_dataloader import build_val_loader
        self.cfg     = cfg
        self.loaders: Dict[Tuple[str, str], object] = {}
        if val_files:
            self.loaders[("ALL", "ALL")] = build_val_loader(val_files, cfg)
        log.info(f"  [ValRegistry] {len(val_files)} files in aggregate loader.")

@torch.no_grad()
def run_validation(
    model: PyannoteSegWrapper,
    val_registry: ValRegistry,
    cfg: FinetuneConfig,
    phase: PhaseConfig,
    device: torch.device,
    device_type: str,
    ecapa,
    epoch: int,
) -> Tuple[float, Dict]:

    model.eval()
    powerset = model.powerset()
    model_max_spk = model._max_spk_per_chunk
    model_max_frame_spk = model._max_spk_per_frame

    all_results: Dict[str, Dict] = {}
    total_val_loss = 0.0
    total_val_der = 0.0

    amp_dtype = getattr(cfg, "amp_dtype", "float16")
    use_non_blocking = device_type == "cuda"

    for (src, spk), loader in val_registry.loaders.items():
        split_name = f"{src}/{spk}"
        split_loss = 0.0
        n_samples_total = 0
        der_acc = {"miss": 0.0, "fa": 0.0, "conf": 0.0, "total": 0.0}
        n_val_crops = 0
        max_crops = getattr(cfg, "max_val_crops", 0)
        invalid_checked = 0

        for batch in tqdm(loader, desc=f"   | Val ({split_name})", leave=False, dynamic_ncols=True):
            waveform = batch["waveform"].to(device, non_blocking=use_non_blocking)
            labels = batch["labels"].to(device, non_blocking=use_non_blocking)
            n_frames = batch["n_frames"].to(device, non_blocking=use_non_blocking)
            load_spk = batch.get("load_speakers", batch.get("n_speakers", None))

            with _autocast(cfg.fp16 and device_type == "cuda", device_type, amp_dtype):
                logits = model(waveform)

            if n_val_crops == 0:
                log.info(f"   [val_debug] wav  min={waveform.min():.4f} max={waveform.max():.4f}")
                log.info(f"   [val_debug] logits shape={logits.shape} min={logits.min():.4f} max={logits.max():.4f}")

            lbl_r, nf_r = _sync_labels_to_output_grid(labels, n_frames, logits.shape[1])
            check_stats = _check_task_compatible_batch(
                lbl_r,
                nf_r,
                model_max_spk,
                model_max_frame_spk,
                policy="ignore",
            )
            invalid_checked += check_stats["invalid"]

            with _autocast(cfg.fp16 and device_type == "cuda", device_type, amp_dtype):
                loss = powerset_pit_loss(
                    logits, lbl_r, nf_r, phase, powerset, model_max_spk, model_max_frame_spk
                )

            n_samples = waveform.shape[0]
            split_loss += loss.item() * n_samples
            n_samples_total += n_samples

            logits_fp32 = logits.detach().cpu().float()
            lbl_der = lbl_r.detach().cpu().float()
            nf_der = nf_r.detach().cpu()
            load_spk_cpu = load_spk.detach().cpu() if torch.is_tensor(load_spk) else None
            m = pit_der_batch(logits_fp32, lbl_der, nf_der, powerset, load_spk_cpu)
            der_acc["miss"] += m["miss_frames"]
            der_acc["fa"] += m["fa_frames"]
            der_acc["conf"] += m["conf_frames"]
            der_acc["total"] += m["total_frames"]

            n_val_crops += 1
            if max_crops > 0 and n_val_crops >= max_crops:
                break

        avg_split_loss = split_loss / max(n_samples_total, 1)
        d = max(der_acc["total"], 1.0)
        der = (der_acc["miss"] + der_acc["fa"] + der_acc["conf"]) / d * 100.0
        miss = der_acc["miss"] / d * 100.0
        fa = der_acc["fa"] / d * 100.0
        conf = der_acc["conf"] / d * 100.0

        log.info(
            f"  [val] {split_name} | loss={avg_split_loss:.4f} | "
            f"DER={der:.2f}% (M:{miss:.2f}% F:{fa:.2f}% C:{conf:.2f}%) | "
            f"decode=argmax | invalid_checked={invalid_checked}"
        )

        all_results[split_name] = {
            "loss": avg_split_loss,
            "der": der,
            "miss": miss,
            "fa": fa,
            "conf": conf,
            "decode_mode": "argmax_powerset",
            "invalid_checked": invalid_checked,
        }
        if src == "ALL" and spk == "ALL":
            total_val_loss = avg_split_loss
            total_val_der = der

    return total_val_loss, {
        "val_loss": total_val_loss,
        "val_der": total_val_der,
        "val_decode_mode": "argmax_powerset",
        "val_splits": all_results,
    }

def _resample_probs_to_frame_grid(probs: np.ndarray, actual_sec: float, dt_sec: float) -> np.ndarray:
    target_frames = max(1, int(round(actual_sec / dt_sec)))
    if probs.shape[0] == target_frames:
        return probs.astype(np.float32, copy=False)
    x = torch.from_numpy(probs.T).unsqueeze(0).float()
    y = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
    return y.squeeze(0).T.numpy().astype(np.float32, copy=False)

def _align_chunk_probs(existing: np.ndarray, current: np.ndarray) -> np.ndarray:
    n_spk = existing.shape[1]
    if existing.shape[0] == 0 or current.shape[0] == 0:
        return np.arange(n_spk, dtype=int)
    T    = min(existing.shape[0], current.shape[0])
    cost = np.array([
        [np.mean(np.abs(existing[:T, i] - current[:T, j])) for j in range(n_spk)]
        for i in range(n_spk)
    ], dtype=np.float32)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.arange(n_spk, dtype=int)
    for i, j in zip(row_ind, col_ind):
        perm[i] = j
    return perm

def _binary_matrix_to_annotation(binary: np.ndarray, uri: str, dt_sec: float = FRAME_SHIFT_SEC, min_dur_sec: float = 0.10):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    track_id = 0
    T, S = binary.shape
    for s in range(S):
        active = binary[:, s].astype(bool)
        in_seg = False; seg_start = 0
        for f in range(T + 1):
            is_on = (f < T) and active[f]
            if is_on and not in_seg:
                seg_start = f; in_seg = True
            elif not is_on and in_seg:
                in_seg = False
                t0 = seg_start * dt_sec; t1 = f * dt_sec
                if t1 - t0 >= min_dur_sec:
                    ann[Segment(t0, t1), f"H{track_id:06d}"] = f"SPK_{s:02d}"
                    track_id += 1
    return ann

def _reference_annotation_from_rttm(sample: Sample):
    from pyannote.core import Annotation, Segment
    ref = Annotation(uri=sample.wav_path)
    for idx, seg in enumerate(parse_rttm(sample.rttm_path)):
        ref[Segment(seg["start"], seg["end"]), f"R{idx:06d}"] = seg["speaker"]
    return ref

def _metric_breakdown(metric) -> Dict[str, float]:
    der_pct = float(abs(metric)) * 100.0
    miss_pct = fa_pct = conf_pct = -1.0
    try:
        detail = metric.report(display=False)
        def _find_col(df, name):
            for col in df.columns:
                label = col[-1] if isinstance(col, tuple) else col
                if isinstance(label, str) and label.lower() == name.lower():
                    return col
            return None
        last = detail.iloc[-1]
        c_total = _find_col(detail, "total")
        c_miss  = _find_col(detail, "missed detection")
        c_fa    = _find_col(detail, "false alarm")
        c_conf  = _find_col(detail, "confusion")
        if c_total is not None:
            total_dur = float(last[c_total])
            if total_dur > 0:
                if c_miss  is not None: miss_pct  = float(last[c_miss])  / total_dur * 100.0
                if c_fa    is not None: fa_pct    = float(last[c_fa])    / total_dur * 100.0
                if c_conf  is not None: conf_pct  = float(last[c_conf])  / total_dur * 100.0
    except Exception as e:
        log.warning(f"  [heavy_valid] metric breakdown failed: {e}")
    return {"der": der_pct, "miss": miss_pct, "fa": fa_pct, "conf": conf_pct}

@torch.no_grad()
def heavy_file_level_der_eval(
    model: PyannoteSegWrapper,
    samples: List[Sample],
    cfg: FinetuneConfig,
    device: torch.device,
    device_type: str,
    threshold: float = 0.5,
    collar: float = 0.25,
    split_name: str = "heavy",
) -> Dict[str, float]:
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        log.warning("[heavy_valid] pyannote.metrics not installed — skipping.")
        return {"der": -1.0, "miss": -1.0, "fa": -1.0, "conf": -1.0, "n_files": 0}

    powerset = model.powerset()
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    chunk_smp = int(cfg.chunk_sec * SR)
    hop_smp = max(1, int(getattr(cfg, "heavy_chunk_hop_sec", cfg.chunk_sec) * SR))
    hop_smp = min(hop_smp, chunk_smp)
    n_files_ok = 0
    skipped_gt_eval_cap = 0
    eval_cap = int(getattr(cfg, "eval_max_total_speakers", getattr(cfg, "max_speakers", 4)))

    for sample in samples:
        if sample.n_speakers > eval_cap:
            skipped_gt_eval_cap += 1
            continue
        try:
            wav, sr = _load_audio(sample.wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
        except Exception as e:
            log.warning(f"  [heavy_valid/{split_name}] load failed {sample.wav_path}: {e}")
            continue

        total_smp = wav.shape[1]
        total_frames = max(1, int(round(total_smp / SR / FRAME_SHIFT_SEC)))
        n_spk_model = model._max_spk_per_chunk
        sum_probs = np.zeros((total_frames, n_spk_model), dtype=np.float32)
        cnt_probs = np.zeros((total_frames, 1), dtype=np.float32)

        for start in range(0, total_smp, hop_smp):
            end = min(start + chunk_smp, total_smp)
            actual_smp = end - start
            if actual_smp < SR:
                break
            chunk = wav[:, start:end]
            if actual_smp < chunk_smp:
                chunk = F.pad(chunk, (0, chunk_smp - actual_smp))
            chunk = chunk.to(device, non_blocking=(device_type == "cuda"))

            amp_dtype = getattr(cfg, "amp_dtype", "float16")
            with _autocast(cfg.fp16 and device_type == "cuda", device_type, amp_dtype):
                logits = model(chunk.unsqueeze(0))[0].detach().cpu().float()

            probs = decode_probs_argmax(logits, powerset).numpy().astype(np.float32)

            actual_frames_model = max(1, int(round(actual_smp / chunk_smp * probs.shape[0])))
            probs = probs[:actual_frames_model]
            probs = _resample_probs_to_frame_grid(probs, actual_smp / SR, FRAME_SHIFT_SEC)

            start_fr = int(round(start / SR / FRAME_SHIFT_SEC))
            end_fr = min(total_frames, start_fr + probs.shape[0])
            probs = probs[:max(0, end_fr - start_fr)]
            if probs.shape[0] == 0:
                continue

            ov_start = start_fr
            ov_end = min(end_fr, start_fr + int(round((chunk_smp - hop_smp) / SR / FRAME_SHIFT_SEC)))
            if ov_end > ov_start and np.any(cnt_probs[ov_start:ov_end, 0] > 0):
                existing = sum_probs[ov_start:ov_end] / np.clip(cnt_probs[ov_start:ov_end], 1e-8, None)
                current = probs[:ov_end - ov_start]
                perm = _align_chunk_probs(existing, current)
                probs = probs[:, perm]

            sum_probs[start_fr:end_fr] += probs
            cnt_probs[start_fr:end_fr] += 1.0

        if np.all(cnt_probs == 0):
            log.warning(f"  [heavy_valid/{split_name}] no coverage: {sample.wav_path}")
            continue

        avg_probs = sum_probs / np.clip(cnt_probs, 1e-8, None)
        hyp_bin = (avg_probs > 0.5).astype(np.uint8)
        hyp = _binary_matrix_to_annotation(hyp_bin, uri=sample.wav_path)
        ref = _reference_annotation_from_rttm(sample)
        if len(ref) == 0:
            continue
        try:
            metric(ref, hyp)
            n_files_ok += 1
        except Exception as e:
            log.warning(f"  [heavy_valid/{split_name}] eval failed {sample.wav_path}: {e}")

    if n_files_ok == 0:
        return {
            "der": -1.0, "miss": -1.0, "fa": -1.0, "conf": -1.0,
            "n_files": 0, "skipped_gt_eval_cap": skipped_gt_eval_cap,
            "eval_max_total_speakers": eval_cap, "decode_mode": "argmax_powerset_stitched"
        }

    out = _metric_breakdown(metric)
    out.update({
        "n_files": n_files_ok,
        "skipped_gt_eval_cap": skipped_gt_eval_cap,
        "eval_max_total_speakers": eval_cap,
        "decode_mode": "argmax_powerset_stitched",
    })
    return out

@torch.no_grad()
def run_heavy_validation(
    model: PyannoteSegWrapper,
    heavy_splits: Dict[str, List[Sample]],
    cfg: FinetuneConfig,
    device: torch.device,
    device_type: str,
    epoch: int,
) -> Dict[str, Dict]:
    if not getattr(cfg, "heavy_validate_enabled", False):
        return {}
    out = {}
    log.info(f"  [heavy_valid] epoch={epoch} | decode=argmax_powerset_stitched")
    for split_name, samples in heavy_splits.items():
        if not samples:
            out[split_name] = {"der": -1.0, "miss": -1.0, "fa": -1.0, "conf": -1.0, "n_files": 0}
            continue
        res = heavy_file_level_der_eval(model, samples, cfg, device, device_type, split_name=split_name)
        out[split_name] = res
        log.info(
            f"  [heavy_valid/{split_name}] n_files={res['n_files']} | "
            f"DER={res['der']:.2f}% (M:{res['miss']:.2f}% F:{res['fa']:.2f}% C:{res['conf']:.2f}%)"
        )
    return out

def train_phase(
    model: PyannoteSegWrapper,
    train_files: List[Sample],
    val_registry: ValRegistry,
    heavy_splits: Dict[str, List[Sample]],
    cfg: FinetuneConfig,
    phase: PhaseConfig,
    device: torch.device,
    device_type: str,
    ecapa,
    out_dir: Path,
    global_epoch: int,
    augmenter: Optional[AudioAugmenter],
) -> Tuple[int, float, List[Dict], List[Dict]]:

    log.info(f"\n{'═'*70}")
    log.info(f"  PHASE: {phase.name}  ({phase.epochs} epochs, freeze={phase.freeze_strategy})")
    log.info(f"{'═'*70}")

    apply_freeze(model, phase.freeze_strategy, phase.unfreeze_top_n)
    optimizer = build_optimizer(model, phase)

    from ft_dataloader import sliding_window_samples
    _mix0        = phase.data_mix.get(0)
    _active      = [s for s in train_files if _mix0.get(s.source, 0.0) > 0.0] or train_files
    _phase_stride = cfg.train_stride_sec_p3 if "phase3" in phase.name.lower() else cfg.train_stride_sec
    _crops0      = sliding_window_samples(_active, cfg.chunk_sec, _phase_stride)

    _micro_steps = max(len(_crops0) // cfg.batch_size, 1) * phase.epochs
    total_steps  = max(_micro_steps // max(cfg.grad_accum, 1), 1)
    del _crops0, _active

    scheduler        = WarmupCosine(optimizer, warmup=phase.warmup_steps, total=total_steps)
    amp_dtype        = getattr(cfg, "amp_dtype", "float16")
    scaler           = _make_scaler(cfg.fp16, device_type, amp_dtype)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_non_blocking = device_type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")
    best_val_der  = float("inf")
    patience_cnt  = 0
    history       = []
    heavy_history = []

    powerset      = model.powerset()
    model_max_spk = model._max_spk_per_chunk
    model_max_frame_spk = model._max_spk_per_frame

    for ep_idx in range(phase.epochs):
        global_epoch += 1
        phase_stride  = cfg.train_stride_sec_p3 if "phase3" in phase.name.lower() else cfg.train_stride_sec
        loader = build_train_loader(
            train_files, phase, ep_idx, cfg,
            augmenter if phase.use_augmentation else None,
            phase_stride,
        )

        model.train()
        epoch_loss = 0.0; n_samples_train = 0; n_steps = 0

        log.info(f"\n{'─'*60}")
        log.info(f"  [{phase.name}] Epoch {ep_idx+1}/{phase.epochs}  (global ep {global_epoch})")
        log.info(f"  mix: {phase.data_mix.get(ep_idx)}")
        log.info(f"{'─'*60}")

        pbar = tqdm(loader, desc=f"Ep {ep_idx+1}/{phase.epochs}", leave=True, ncols=70, ascii=True, miniters=1, mininterval=1.0)
        for step, batch in enumerate(pbar):
            waveform = batch["waveform"].to(device, non_blocking=use_non_blocking)
            labels   = batch["labels"].to(device, non_blocking=use_non_blocking)
            n_frames = batch["n_frames"].to(device, non_blocking=use_non_blocking)

            with _autocast(cfg.fp16 and device_type == "cuda", device_type, amp_dtype):
                logits = model(waveform)

            lbl_r, nf_r = _sync_labels_to_output_grid(labels, n_frames, logits.shape[1])

            _check_task_compatible_batch(
                lbl_r,
                nf_r,
                model_max_spk,
                model_max_frame_spk,
                policy="ignore",
            )

            with _autocast(cfg.fp16 and device_type == "cuda", device_type, amp_dtype):
                loss = powerset_pit_loss(
                    logits, lbl_r, nf_r, phase, powerset, model_max_spk, model_max_frame_spk
                ) / cfg.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, phase.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss      += loss.item() * cfg.grad_accum * waveform.shape[0]
            n_samples_train += waveform.shape[0]
            n_steps         += 1
            pbar.set_postfix(loss=f"{loss.item()*cfg.grad_accum:.4f}", lr=f"{scheduler.get_lr()[0]:.2e}")

        remainder = n_steps % cfg.grad_accum
        if remainder > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, phase.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        avg_loss = epoch_loss / max(n_samples_train, 1)
        log.info(f"\n  avg_train_loss={avg_loss:.4f}")

        val_metrics: Dict = {}
        do_val = (ep_idx + 1) % cfg.val_every_n_epochs == 0 or (ep_idx + 1) == phase.epochs
        if do_val:
            val_loss, val_metrics = run_validation(
                model, val_registry, cfg, phase, device, device_type, ecapa, global_epoch
            )
            val_der = val_metrics.get("val_der", float("inf"))
            if val_der < best_val_der:
                best_val_der  = val_der
                best_val_loss = val_loss
                patience_cnt  = 0
                phase_best_path = str(out_dir / f"best_model_{phase.name}.pth")
                best_path = str(out_dir / "best_model.pth")

                clean_sd = {
                    k[len("_seg_model."):] if k.startswith("_seg_model.") else k: v
                    for k, v in model.state_dict().items()
                }
                phase_payload = {
                    "epoch": global_epoch,
                    "phase": phase.name,
                    "val_loss": val_loss, "val_der": val_der,
                    "val_decode_mode": "argmax_powerset",
                    "model_state": clean_sd,
                    "max_speakers_per_chunk": model._max_spk_per_chunk,
                    "max_speakers_per_frame": model._max_spk_per_frame,
                    "eval_max_total_speakers": getattr(cfg, "eval_max_total_speakers", 4),
                }
                torch.save(phase_payload, phase_best_path)

                should_update_global = True
                if Path(best_path).exists():
                    try:
                        prev_best = torch.load(best_path, map_location="cpu", weights_only=False)
                        prev_der = float(prev_best.get("val_der", float("inf")))
                        should_update_global = val_der < prev_der
                    except Exception:
                        should_update_global = True
                if should_update_global:
                    torch.save(phase_payload, best_path)
                    log.info(f"  ★ New GLOBAL best val_der={val_der:.2f}% (loss={val_loss:.4f}) → {best_path}")
                else:
                    log.info(f"   Phase best val_der={val_der:.2f}% saved → {phase_best_path} (global best unchanged)")
            else:
                patience_cnt += 1
                log.info(f"  No improvement. Patience: {patience_cnt}/{cfg.patience}")
                if patience_cnt >= cfg.patience:
                    log.info(f"  Early stopping at epoch {global_epoch}")
                    break

        heavy_metrics: Dict = {}
        do_heavy = (
            getattr(cfg, "heavy_validate_enabled", False)
            and bool(heavy_splits)
            and (global_epoch % max(1, int(getattr(cfg, "heavy_validate_every", 3))) == 0
                 or (ep_idx + 1) == phase.epochs)
        )
        if do_heavy:
            heavy_metrics = run_heavy_validation(model, heavy_splits, cfg, device, device_type, global_epoch)
            heavy_history.append({
                "global_epoch": global_epoch, "phase": phase.name,
                "decode_mode": "argmax_powerset_stitched",
                "splits": heavy_metrics,
            })

        if (ep_idx + 1) % cfg.save_every_n_epochs == 0:
            clean_sd = {
                k[len("_seg_model."):] if k.startswith("_seg_model.") else k: v
                for k, v in model.state_dict().items()
            }
            ep_path = str(out_dir / f"ep{global_epoch:03d}_{phase.name}.pth")
            torch.save({
                "epoch": global_epoch, "phase": phase.name,
                "train_loss": avg_loss, **val_metrics,
                "heavy_valid": heavy_metrics, "model_state": clean_sd,
            }, ep_path)
            log.info(f"  [ckpt] {ep_path}")

        history.append({
            "global_epoch": global_epoch, "phase": phase.name,
            "train_loss": avg_loss, **val_metrics, "heavy_valid": heavy_metrics,
        })

    return global_epoch, best_val_der, history, heavy_history

def main():
    import random
    cfg = FinetuneConfig()

    if TEST_MODE:
        MAX_N_FILES = 2
        log.info("=" * 60)
        log.info("  MODE: SMOKE TEST  (TEST_MODE=True trong ft_config.py)")
        log.info("=" * 60)
    else:
        MAX_N_FILES = None
        log.info("=" * 60)
        log.info("  MODE: FULL TRAINING  (TEST_MODE=False)")
        log.info("=" * 60)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    device_type = _get_device_type(device)
    if cfg.require_cuda and device_type != "cuda":
        log.error("CUDA is required. Refusing to fall back to MPS/CPU.")
        raise SystemExit(1)
    log.info(f"Device: {device}")

    if device_type == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(getattr(cfg, "matmul_precision", "high")))
        if getattr(cfg, "enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if getattr(cfg, "cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        log.info(
            f"  [CUDA] amp_dtype={getattr(cfg,'amp_dtype','float16')} | "
            f"TF32={getattr(cfg,'enable_tf32',True)} | "
            f"cudnn_benchmark={getattr(cfg,'cudnn_benchmark',True)}"
        )

    if device_type == "mps" and cfg.fp16:
        log.warning("  [!] MPS detected. Disabling AMP FP16.")
        cfg.fp16 = False

    log.info("\n[Data] Scanning...")
    train_files = scan_all_sources(cfg, max_n_files_per_source=MAX_N_FILES)
    if not train_files:
        log.error("No training files found!"); sys.exit(1)

    for s in train_files:
        log.info(
            f"  [train/{s.source}] {Path(s.wav_path).parent.parent.name}"
            f" | dur={s.duration:.1f}s | n_spk={s.n_speakers}"
        )

    from pathlib import Path as _P
    val_files_labeled = scan_fixed_split_dir(
        VAL_LABELED_DIR, str(_P(VAL_LABELED_DIR) / "metadata.txt"),
        max_speakers=getattr(cfg, "eval_max_total_speakers", cfg.max_speakers), source_name="val_labeled",
    )
    val_files_datamix = scan_fixed_split_dir(
        VAL_DATA_DIR, str(_P(VAL_DATA_DIR) / "metadata.txt"),
        max_speakers=getattr(cfg, "eval_max_total_speakers", cfg.max_speakers), source_name="val_datamix",
    )
    from ft_dataloader import stratified_sample_by_overlap, stratified_sample_by_overlap_and_speakers
    if len(val_files_labeled) >= cfg.max_val_files:
        val_files = stratified_sample_by_overlap_and_speakers(val_files_labeled, cfg.max_val_files)
    else:
        needed    = cfg.max_val_files - len(val_files_labeled)
        val_files = val_files_labeled + stratified_sample_by_overlap(val_files_datamix, needed)
    if not val_files:
        log.warning("[Val] No val files — using train subset")
        val_files = list(train_files[:cfg.max_val_files])
    log.info(f"\n[Val] {len(val_files)} files (labeled={len(val_files_labeled)})")

    test_files_labeled = scan_fixed_split_dir(
        TEST_LABELED_DIR, str(_P(TEST_LABELED_DIR) / "metadata.txt"),
        max_speakers=getattr(cfg, "eval_max_total_speakers", cfg.max_speakers), source_name="test_labeled",
    )
    test_files_datamix = scan_fixed_split_dir(
        TEST_DATA_DIR, str(_P(TEST_DATA_DIR) / "metadata.txt"),
        max_speakers=getattr(cfg, "eval_max_total_speakers", cfg.max_speakers), source_name="test_datamix",
    )
    test_files = test_files_labeled + test_files_datamix
    if not test_files:
        log.warning("[Test] No test files — using val files for final eval")
        test_files = list(val_files)
    log.info(f"[Test] {len(test_files)} files")

    augmenter = AudioAugmenter(sr=SR)

    val_registry = ValRegistry(val_files, cfg)

    def _limit_heavy(files):
        eval_cap = int(getattr(cfg, "eval_max_total_speakers", cfg.max_speakers))
        keep = [s for s in files if s.n_speakers <= eval_cap]
        mx = int(getattr(cfg, "heavy_max_files_per_split", 0))
        return keep[:mx] if mx > 0 else keep

    heavy_splits = {
        "heavy_labeled": _limit_heavy(val_files_labeled),
        "heavy_datamix": _limit_heavy(val_files_datamix),
    }
    for _name, _files in heavy_splits.items():
        log.info(f"[Heavy] {_name}={len(_files)} files")

    ecapa = load_ecapa_model()

    log.info(f"\n[Model] Loading: {cfg.pretrained_model}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token is None:
        log.warning(
            "  [!] HF_TOKEN không được set. "
            "Nếu lỗi authentication, hãy:\n"
            "    export HF_TOKEN=hf_xxxxxxxxxx\n"
            "  hoặc: huggingface-cli login"
        )

    try:
        from pyannote.audio import Pipeline

        safe_globals = []
        try:
            safe_globals.append(torch.torch_version.TorchVersion)
        except AttributeError:
            log.warning("  [safe_globals] torch.torch_version.TorchVersion không tìm thấy — bỏ qua.")
        try:
            from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
            safe_globals.extend([Problem, Resolution, Specifications, Task])
        except Exception:
            pass
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_globals)

        try:
            from pyannote.pipeline.pipeline import Pipeline as PyaPipeline
            _original_instantiate = PyaPipeline.instantiate
            def _safe_instantiate(self, parameters):
                try:
                    return _original_instantiate(self, parameters)
                except ValueError as e:
                    log.warning(f"  [patch] Ignored instantiate error: {e}")
            PyaPipeline.instantiate = _safe_instantiate
        except Exception as e:
            log.debug(f"  [patch] Could not monkey-patch Pipeline.instantiate: {e}")

        try:
            from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
            _original_init = SpeakerDiarization.__init__

            def _init_without_plda(self, *args, **kwargs):

                if 'plda' in kwargs:
                    log.debug(f"  [patch] Removing 'plda' parameter from {self.__class__.__name__}")
                    kwargs.pop('plda')
                log.info(f"SpeakerDiarization kwargs: {kwargs}")

                if kwargs.get('segmentation') == '$model/segmentation':
                    log.info("Detected '$model/segmentation', replacing with 'pyannote/segmentation-3.0'")
                    kwargs['segmentation'] = 'pyannote/segmentation-3.0'

                if kwargs.get('embedding') == '$model/embedding':
                    log.info("Detected '$model/embedding', replacing with 'speechbrain/spkrec-ecapa-voxceleb'")
                    kwargs['embedding'] = 'speechbrain/spkrec-ecapa-voxceleb'

                if kwargs.get('clustering') == 'VBxClustering':
                    log.info("Detected 'VBxClustering', replacing with 'AgglomerativeClustering'")
                    kwargs['clustering'] = 'AgglomerativeClustering'

                return _original_init(self, *args, **kwargs)

            SpeakerDiarization.__init__ = _init_without_plda
        except Exception as e:
            log.debug(f"  [patch] Could not monkey-patch SpeakerDiarization: {e}")

        pipeline = None
        try:

            pipeline = Pipeline.from_pretrained(
                cfg.pretrained_model,
                use_auth_token=hf_token if hf_token else True,
            )
            log.info("  pyannote Pipeline  (with auth_token)")
        except Exception as e1:
            error_msg = str(e1)
            log.debug(f"  [pipeline] First attempt failed: {type(e1).__name__}: {error_msg[:100]}")

            try:
                pipeline = Pipeline.from_pretrained(cfg.pretrained_model)
                log.info("  pyannote Pipeline  (without auth_token param)")
            except Exception as e2:
                import traceback
                traceback.print_exc()
                log.error(f"Failed to load pipeline: {type(e2).__name__}: {e2}")
                log.error(
                    "  Kiểm tra:\n"
                    "  1. Đã accept điều khoản model trên HuggingFace chưa?\n"
                    "     https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                    "  2. HF_TOKEN đã set chưa? (export HF_TOKEN=hf_xxx)\n"
                    "  3. pip install pyannote.audio (nếu chưa)\n"
                    "  4. pretrained_model trong ft_config.py có đúng không?\n"
                    "     → pretrained_model = 'pyannote/speaker-diarization-community-1'"
                )
                raise SystemExit(1)

    except Exception as e:
        log.error(f"Failed to load pipeline: {type(e).__name__}: {e}")
        log.error(
            "  Kiểm tra:\n"
            "  1. Đã accept điều khoản model trên HuggingFace chưa?\n"
            "     https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2. HF_TOKEN đã set chưa? (export HF_TOKEN=hf_xxx)\n"
            "  3. pip install pyannote.audio (nếu chưa)\n"
            "  4. pretrained_model trong ft_config.py có đúng không?\n"
            "     → pretrained_model = 'pyannote/speaker-diarization-community-1'"
        )
        raise SystemExit(1)

    model_chunk_cap = int(getattr(cfg, "model_max_speakers_per_chunk", 3))
    expected_frame_cap = int(getattr(cfg, "model_max_speakers_per_frame", 2))
    model = PyannoteSegWrapper(pipeline, max_speakers=model_chunk_cap).to(device)

    if model_chunk_cap != model._max_spk_per_chunk:
        log.warning(
            f"  [!] cfg.model_max_speakers_per_chunk={model_chunk_cap} but loaded model.spec.max_speakers_per_chunk={model._max_spk_per_chunk}. "
            f"Using loaded model value={model._max_spk_per_chunk}."
        )
    if expected_frame_cap != model._max_spk_per_frame:
        log.warning(
            f"  [!] cfg.model_max_speakers_per_frame={expected_frame_cap} but loaded model.spec.max_speakers_per_frame={model._max_spk_per_frame}. "
            f"Using loaded model value={model._max_spk_per_frame}."
        )

    if device_type == "cuda" and getattr(cfg, "compile_model", False) and hasattr(torch, "compile"):
        try:
            model._seg_model = torch.compile(
                model._seg_model,
                mode=str(getattr(cfg, "compile_mode", "max-autotune")),
                dynamic=True,
            )
            log.info(f"  [CUDA] torch.compile enabled: mode={getattr(cfg, 'compile_mode', 'max-autotune')}")
        except Exception as e:
            log.warning(f"  [CUDA] torch.compile failed: {e}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_dir}")

    phase_configs = build_phase_configs()
    global_epoch  = 0
    full_history  = []
    heavy_history = []
    best_ckpt_path = str(out_dir / "best_model.pth")

    for phase in phase_configs:
        global_epoch, phase_best_der, phase_history, phase_heavy = train_phase(
            model, train_files, val_registry, heavy_splits, cfg, phase,
            device, device_type, ecapa, out_dir, global_epoch, augmenter,
        )
        full_history.extend(phase_history)
        heavy_history.extend(phase_heavy)
        log.info(f"\n  [Phase done] {phase.name} | best_val_der={phase_best_der:.2f}%")

    log.info(f"\n{'═'*70}")
    log.info("TRAINING COMPLETE")
    log.info(f"{'═'*70}")
    for r in full_history:
        val_loss = r.get("val_loss", -1.0)
        ep_str   = f"ep{r['global_epoch']:>3} [{r['phase']:<20}] train_loss={r['train_loss']:.4f}"
        log.info(ep_str + (f"  val_loss={val_loss:.4f}" if val_loss >= 0 else "  (no val)"))

    out_json = str(out_dir / "training_history.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2, ensure_ascii=False)
    log.info(f"\nHistory saved → {out_json}")

    heavy_json = str(out_dir / "heavy_validation_history.json")
    with open(heavy_json, "w", encoding="utf-8") as f:
        json.dump(heavy_history, f, indent=2, ensure_ascii=False)
    log.info(f"Heavy history saved → {heavy_json}")

    if Path(best_ckpt_path).exists() and test_files:
        log.info(f"\n{'═'*70}")
        log.info("  FINAL EVALUATION  (DER@files<=4spk)")
        log.info(f"{'═'*70}")

        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model_state = ckpt["model_state"]

        missing, unexpected = model._seg_model.load_state_dict(model_state, strict=False)
        if missing:
            log.warning(f"  [final] Missing keys: {missing[:5]}")
        if unexpected:
            log.warning(f"  [final] Unexpected keys: {unexpected[:5]}")
        model.eval()

        final_test_splits = {
            "test_labeled": [s for s in test_files_labeled if s.n_speakers <= getattr(cfg, "eval_max_total_speakers", cfg.max_speakers)],
            "test_datamix": [s for s in test_files_datamix if s.n_speakers <= getattr(cfg, "eval_max_total_speakers", cfg.max_speakers)],
        }
        final_results = {}
        for split_name, split_files in final_test_splits.items():
            if not split_files:
                log.info(f"  [final/{split_name}] skipped (0 files)")
                final_results[split_name] = {"der": -1.0, "n_files": 0}
                continue
            res = heavy_file_level_der_eval(
                model, split_files, cfg, device, device_type,
                collar=0.25, split_name=split_name,
            )
            final_results[split_name] = res
            log.info(
                f"  [final/{split_name}] n_files={res['n_files']} | "
                f"DER={res['der']:.2f}% (M:{res['miss']:.2f}% F:{res['fa']:.2f}% C:{res['conf']:.2f}%)"
            )

        final_path = str(out_dir / "final_evaluation.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        log.info(f"Final evaluation saved → {final_path}")

if __name__ == "__main__":
    main()