"""vision cell for v2 (adaptive MedSigLIP encoder, one encode() interface)."""

# ---- vision: adaptive encoder (HF SigLIP | KerasHub MedSigLIP | stub) ----
# Fast-path contract: when a frozen encoder is present, every slot image is
# encoded ONCE into a (studies, N_SLOT, D) float32 feature cache and the
# image caches are freed; the head then trains on precomputed features,
# which is what makes the GPU part finish in minutes instead of hours.

import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # before any keras import

VISION_MICROBATCH = 8

_INPUT_SCAN = None


def kaggle_input_dirs():
    """One memoised walk of the input root; the scan is otherwise the
    single most wasteful thing on Kaggle (rglob is revisited everywhere)."""
    global _INPUT_SCAN
    if _INPUT_SCAN is None:
        base = Path("/kaggle/input") if Path("/kaggle/input").is_dir() \
            else Path(".")
        _INPUT_SCAN = {str(r) for r in base.rglob("*") if r.is_dir()}
    return _INPUT_SCAN


class VisionEncoder(nn.Module):
    """interface: encode(x [B,C,H,W] in [0,1]) -> (B,D); feature_dim: int"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.frozen = True              # frozen encoders unlock the feat cache
        self.grad = False               # True = finetune mode (encoder in train)

    def set_grad(self, trainable_blocks=0):
        """Unfreeze the last trainable_blocks transformer blocks + the
        post-encoder projection, leave the rest frozen. Returns the number
        of trainable parameters (>0 means fine-tuning is live)."""
        self.grad = trainable_blocks > 0
        if self.grad:
            self.frozen = False
        n = 0
        for p in self.parameters():
            if p.requires_grad:
                n += p.numel()
        return n

    def encode(self, x):
        raise NotImplementedError


class HFEncoder(VisionEncoder):
    def __init__(self, model, feature_dim, device):
        super().__init__(feature_dim)
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _blocks(self):
        """-> (vision stream, list of transformer blocks) or (None, [])."""
        m = self.model
        if hasattr(m, "vision_model"):
            vs = m.vision_model
        elif hasattr(m, "vision_encoder"):
            vs = m.vision_encoder
        else:
            vs = m
        if hasattr(vs, "encoder") and hasattr(vs.encoder, "layers"):
            return vs, list(vs.encoder.layers)
        return vs, []

    def set_grad(self, trainable_blocks=0):
        for p in self.model.parameters():
            p.requires_grad_(False)
        vs, blocks = self._blocks()
        n = 0
        if trainable_blocks > 0 and len(blocks):
            for b in blocks[-trainable_blocks:]:
                for p in b.parameters():
                    p.requires_grad_(True)
                    n += p.numel()
        # the pooled projection feeds the head; unfreeze with the blocks
        if hasattr(vs, "pooler") and n:
            for p in vs.pooler.parameters():
                p.requires_grad_(True)
                n += p.numel()
        elif hasattr(vs, "post_layernorm") and n:
            for p in vs.post_layernorm.parameters():
                p.requires_grad_(True)
                n += p.numel()
        self.grad = n > 0
        self.frozen = not self.grad
        return n

    def encode(self, x):
        if self.grad:
            self.model.train()
            out = self.model(pixel_values=x).last_hidden_state
        else:
            with torch.no_grad():
                self.model.eval()
                out = self.model(pixel_values=x).last_hidden_state
        return out.mean(1)


class KerasEncoder(VisionEncoder):
    """keras_hub preset behind the same encode() contract (torch backend)."""

    def __init__(self, backbone, converter, feature_dim, device):
        super().__init__(feature_dim)
        self.backbone = backbone
        self.converter = converter
        for layer in getattr(backbone, "_layers", []):
            try:
                layer.trainable = False
            except Exception:
                pass
        try:
            self.backbone.to(device)
        except Exception:
            pass
        self.backbone.eval()
        self.device = device

    def set_grad(self, trainable_blocks=0):
        """KerasEncoder fine-tuning is NOT supported: the keras-hub eager
        graph can't be half-unfrozen without a verified recompile, and
        unfreezing all layers at 448px would blow T4 memory. The HF encoder
        path carries fine-tuning; Keras stays on the frozen feature cache.
        Returns 0 so the caller falls back to the feats path."""
        self.grad = False
        self.frozen = True
        return 0

    def _vision(self, x):
        if hasattr(self.backbone, "vision_encoder"):
            return self.backbone.vision_encoder
        for cand in ("vision", "vision_model", "image_encoder", "visual"):
            if hasattr(self.backbone, cand):
                return getattr(self.backbone, cand)
        raise AttributeError("no vision stream on preset")

    def _pick(self, y):
        if isinstance(y, dict):
            for k in ("sequence_embedding", "pooled_embedding"):
                if k in y and y[k] is not None:
                    return y[k]
            return next(iter(y.values()))
        if hasattr(y, "sequence_embedding"):
            return y.sequence_embedding
        if hasattr(y, "pooled_embedding"):
            return y.pooled_embedding
        return y

    def encode(self, x):
        x = (x * 255.0).byte().permute(0, 2, 3, 1)      # B,H,W,C uint8
        with torch.no_grad():
            pre = self.converter(x)
            try:
                y = self._vision(pre) if not isinstance(pre, dict) else \
                    self._vision({"images": pre["images"]})
            except TypeError:
                y = self._vision({"images": pre if not isinstance(pre, dict)
                                  else pre["images"]})
            if isinstance(y, dict) or hasattr(y, "sequence_embedding"):
                y = self._pick(y)
            t = torch.as_tensor(y, device=self.device)
            if t.dim() == 3:
                t = t.mean(1)
        return t.float()


class StubEncoder(VisionEncoder):
    def __init__(self, dim, device):
        super().__init__(dim)
        self.frozen = False          # trainable: keep the pixel pipeline
        self.model = nn.Sequential(nn.AdaptiveAvgPool2d((32, 32)), nn.Flatten(),
                                   nn.Linear(3 * 32 * 32, dim)).to(device)
        self.device = device

    def set_grad(self, trainable_blocks=0):
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.grad = n > 0
        self.frozen = not self.grad
        return n

    def encode(self, x):
        return self.model(x)


class FeatBackbone(VisionEncoder):
    """Header-only marker for the feature-cache path: the heavy encoder is
    burned into feat_cache_*.npz; models just need the width, not the
    weights (which would otherwise be duplicated per fold)."""

    def __init__(self, feature_dim, device):
        super().__init__(feature_dim)
        self.device = device

    def set_grad(self, trainable_blocks=0):
        return 0                       # features already cached by the caller

    def encode(self, x):
        raise RuntimeError("FeatBackbone is never meant to encode; "
                           "feed precomputed feats instead")

    def to(self, *a, **k):
        return self


def _vision_dirs():
    """config.json-bearing dirs under the input root."""
    out = []
    for s in sorted(kaggle_input_dirs(), key=len):
        if Path(s) / "config.json" is not None and \
                (Path(s) / "config.json").is_file():
            out.append(Path(s))
    return out


def _smoke(enc, device):
    """A zero-image forward at load time; a failed branch must not survive."""
    try:
        dev = next(enc.parameters()).device
    except StopIteration:
        dev = device
    with torch.no_grad():
        y = enc.encode(torch.zeros(2, 3, IMG, IMG, device=dev))
    if y.shape[0] != 2 or y.shape[1] != enc.feature_dim:
        raise ValueError("smoke shape mismatch")
    return enc


def _kerashub_one(d, device):
    """One from_preset attempt; raises on any failure."""
    import keras
    import keras_hub
    keras.config.set_backend("torch")
    backbone = keras_hub.models.Backbone.from_preset(str(d))
    converter = keras_hub.layers.SigLIPImageConverter.from_preset(str(d))
    dim = int(getattr(backbone, "hidden_dim", None) or 1152)
    return KerasEncoder(backbone, converter, dim, device)


def _kerashub_load(d, device):
    """KerasHub preset -> KerasEncoder, with one pip-upgrade retry.

    The Kaggle images keep shipping keras a bit older than the preset's
    serialization; a single upgrade+reimport, gated behind the network
    probe, fixes the DTypePolicy deserialization error in almost all cases.
    """
    import socket, subprocess, sys

    def net_up():
        try:
            socket.create_connection(("pypi.org", 443), timeout=2).close()
            return True
        except Exception:
            return False

    try:
        return _kerashub_one(d, device)
    except Exception as e:
        log(f"vision: KerasHub load failed: {str(e)[:160]}")
    if not net_up():
        raise RuntimeError("keras preset mismatch and no internet for pip")
    try:
        log("vision: upgrading keras/keras-hub (pypi reachable)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--upgrade", "keras", "keras-hub"],
                       capture_output=True, timeout=600)
        for m in list(sys.modules):
            if m == "keras" or m.startswith("keras.") or \
               m == "keras_hub" or m.startswith("keras_hub.") or \
               m == "keras_hub_io" or m.startswith("keras_hub_io."):
                del sys.modules[m]
        try:
            import keras
            import keras_hub
            log(f"keras {keras.__version__} keras_hub {keras_hub.__version__}")
        except Exception as e:
            log(f"vision: keras reimport failed: {e!r}")
        return _kerashub_one(d, device)
    except Exception as e2:
        log(f"vision: KerasHub retry failed: {str(e2)[:160]}")
        raise


def load_vision(device):
    """-> VisionEncoder, never None (last resort = trainable stub).

    HF SiglipVisionModel dirs are tried first (they unlock the same frozen
    feature-cache fast path), then KerasHub presets (with an upgrade retry),
    then the stub - which only trains on pixels and is noticeably slower.
    """
    for d in _vision_dirs():
        try:
            cfg = json.load(open(d / "config.json"))
        except Exception:
            continue
        if cfg.get("model_type") != "siglip":
            continue
        try:
            from transformers import SiglipVisionModel
            m = SiglipVisionModel.from_pretrained(str(d))
            vc = cfg.get("vision_config") or cfg
            dim = vc.get("hidden_size", 768)
            return _smoke(HFEncoder(m, dim, device), device)
        except Exception as e:
            log(f"vision: HF load failed: {e!r}")

    for d in _vision_dirs():
        if not (d / "model.weights.h5").is_file():
            continue
        try:
            return _smoke(_kerashub_load(d, device), device)
        except Exception:
            log(f"vision: KerasHub preset unusable: {str(d)[-60:]}")

    log("vision: no encoder attached - using stub (random init, trainable)")
    return _smoke(StubEncoder(256, device), device)


def build_feature_cache(enc, cache, uids, tag, dev):
    """Encode every cached slot once -> (len(uids), N_SLOT, D) float32.

    The cache lives on disk (feat_cache_<tag>.npz) and is loaded back on
    reruns; on the fast path the caller then drops the image caches and
    trains on the features. Returns None when the encoder is trainable
    (stub) - the pixel pipeline stays in charge in that case.
    """
    if enc is None or getattr(enc, "frozen", False) is False:
        log(f"feats[{tag}]: encoder is trainable - keeping image pipeline")
        return None
    fp = Path(f"feat_cache_{tag}.npz")
    try:
        z = np.load(fp)
        if list(z["uid"]) == list(uids):
            log(f"feats[{tag}]: hit {fp.name} {z['f'].shape}")
            return z["f"].astype(np.float32)
    except Exception:
        pass
    M = len(uids)
    D = enc.feature_dim
    rows = cache[:, :, :GROUP]        # (M, S, 3, H, W) uint8
    f = np.zeros((M, rows.shape[1], D), np.float32)
    step = 16
    enc_dev = torch.device("cpu")
    for p in enc.parameters():
        enc_dev = p.device
        break
    use_amp = enc_dev.type == "cuda"
    for i0 in range(0, M, step):
        batch = rows[i0:i0 + step]
        b, s = batch.shape[:2]
        x = torch.from_numpy(batch).reshape(b * s, *batch.shape[2:]) \
            .float().div_(255.0)
        out = []
        for j in range(0, x.shape[0], VISION_MICROBATCH):
            chunk = x[j:j + VISION_MICROBATCH].to(enc_dev)
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp \
                    else torch.no_grad():
                out.append(enc.encode(chunk))
        f[i0:i0 + step] = torch.cat(out, 0).reshape(b, s, D).cpu().numpy()
        if (i0 // step) % 8 == 0:
            log(f"feats[{tag}]: {i0 + len(batch)}/{M} slots encoded")
    np.savez(fp, uid=np.array(uids, dtype=object), f=f.astype(np.float16))
    log(f"feats[{tag}]: cached {f.shape} -> {fp.name}")
    return f