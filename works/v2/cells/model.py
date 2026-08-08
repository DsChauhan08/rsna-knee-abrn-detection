"""model cell for v2: V2Model = vision + text + fusion -> SlotHead (v1 verbatim)."""

# ---- model: SlotHead (v1) + multimodal fusion ---------------------------

class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study."""

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class V2Model(nn.Module):
    """vision.encode -> (B,S,Dv); report text broadcast per slot;
    concat -> fusion MLP -> SlotHead (v1, untouched)."""

    def __init__(self, vision, text_dim=0, feature_dim=None, fusion_dim=256, p=0.1):
        super().__init__()
        self.vision = vision
        self.text_dim = int(text_dim or 0)
        dv = feature_dim or getattr(vision, "feature_dim", 256)
        self.vision_dim = dv
        self.use_text = self.text_dim > 0
        self.fusion = nn.Sequential(
            nn.Linear(dv + (self.text_dim if self.use_text else 0), fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(p))
        self.head = SlotHead(fusion_dim, N_SLOT, len(TARGETS))

    def encode_slots(self, imgs):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        feats = []
        for i in range(0, x.shape[0], VISION_MICROBATCH):
            chunk = x[i:i + VISION_MICROBATCH]
            if isinstance(self.vision, StubEncoder):
                feats.append(self.vision.encode(chunk))
            elif self.vision.grad:           # fine-tune mode: backprop through
                feats.append(self.vision.encode(chunk))
            else:
                with torch.no_grad():
                    feats.append(self.vision.encode(chunk))
        return torch.cat(feats, 0).reshape(B, S, -1)

    def forward(self, imgs, mask, txt=None, feats=None):
        if feats is None:
            v = self.encode_slots(imgs)
        else:
            v = feats
        if self.use_text:
            if txt is None:
                txt = torch.zeros(v.shape[0], self.text_dim, device=v.device)
            v = torch.cat([v, txt.unsqueeze(1).expand(-1, v.shape[1], -1)], dim=-1)
        return self.head(self.fusion(v), mask)


class Ema:
    """Shadow EMA of trainable params (no deepcopy - safe with Keras encoder).

    skip_prefixes: when fine-tuning the vision stack, EMA tracks only the
    head/fusion (the encoder already averages through its own slow LR);
    shadowing 100M+ encoder params per step would dominate the clock.
    """

    def __init__(self, model, decay, skip_prefixes=()):
        self.decay = decay
        self.step = 0
        self.shadow = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad
            and not n.startswith(skip_prefixes)}

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                s = self.shadow[name].to(p.device)
                s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def target(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.copy_(self.shadow[name].to(p.device))
        return model


def build_v2_model(vision, text_dim=0, feature_dim=None, fusion_dim=256):
    if isinstance(vision, StubEncoder):
        from copy import deepcopy
        vision = deepcopy(vision)     # fresh stub per model (v1 behaviour)
    return V2Model(vision, text_dim=text_dim, feature_dim=feature_dim,
                   fusion_dim=fusion_dim)


def build_eval_model(vision, text_dim, dev, src=None, feat_dim=None):
    """Fresh V2Model; EMA/vision weights copied in (no shared state)."""
    m = build_v2_model(vision, text_dim,
                       feature_dim=feat_dim or getattr(vision, "feature_dim",
                                                       None)).to(dev)
    if src is not None:
        m.load_state_dict(src.state_dict(), strict=False)
    return m