"""text cell for v2: adaptive Gemma-4 loader + one-shot embedding cache."""

# ---- text: Gemma-4 report embeddings (offline, once per study) ----------

TEXT_MAXLEN = 96


def find_model_dir(keys):
    base = Path("/kaggle/input") if Path("/kaggle/input").is_dir() else Path(".")
    best = None
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        if not (d / "config.json").is_file():
            continue
        name = str(d).lower()
        if not any(k in name for k in keys):
            continue
        has_w = (d / "model.safetensors").is_file() or \
                (d / "model.weights.h5").is_file()
        if has_w:
            best = d if best is None else min(best, d, key=lambda p: len(str(p)))
    return best


GEMMA_DIR = find_model_dir(("gemma",))
log(f"Gemma: {str(GEMMA_DIR)}" if GEMMA_DIR else "Gemma NOT attached")


def load_text_model():
    """-> (tokenizer, model) or (None, None). CUDA-only; never raises."""
    if GEMMA_DIR is None:
        return None, None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_DIR))
    except Exception as e:
        log(f"text: tokenizer failed: {e!r}")
        return None, None
    cfg = {}
    try:
        cfg = json.load(open(GEMMA_DIR / "config.json"))
    except Exception:
        pass
    arch = (cfg.get("architectures") or [""])[0]
    try:
        import torch
        try:
            import bitsandbytes  # noqa: F401
            has_bnb = True
        except Exception:
            has_bnb = False
        if has_bnb:
            try:
                if "ConditionalGeneration" in arch:
                    from transformers import AutoModelForMultimodalLM as Klass
                else:
                    from transformers import AutoModelForCausalLM as Klass
                m = Klass.from_pretrained(str(GEMMA_DIR), load_in_4bit=True,
                                          device_map="auto",
                                          torch_dtype=torch.bfloat16)
                log(f"  Gemma loaded 4-bit ({arch or 'auto'})")
                return tokenizer, m
            except Exception as e:
                log(f"  4-bit failed: {str(e)[:160]}")
        try:
            if "ConditionalGeneration" in arch:
                from transformers import AutoModelForMultimodalLM as Klass
            else:
                from transformers import AutoModelForCausalLM as Klass
            m = Klass.from_pretrained(str(GEMMA_DIR), device_map="auto",
                                      torch_dtype=torch.bfloat16)
            log(f"  Gemma loaded bf16 ({arch or 'auto'})")
            return tokenizer, m
        except Exception as e:
            log(f"  bf16 failed: {str(e)[:160]}")
    except Exception as e:
        log(f"  torch/transformers unavailable: {e!r}")
    log("  Gemma unavailable - extractor T0 features carry the text branch")
    return None, None


def text_embed_batch(model, tokenizer, texts, dev):
    import torch.nn.functional as F
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=TEXT_MAXLEN, return_tensors="pt")
    ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    with torch.no_grad():
        if hasattr(model, "language_model"):
            out = model.language_model(input_ids=ids, attention_mask=am,
                                       output_hidden_states=True)
        else:
            out = model(input_ids=ids, attention_mask=am,
                        output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = am.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        pooled = F.normalize(pooled, p=2, dim=-1)
    return pooled.float().cpu().numpy()


def build_text_cache(uids, reports, dev, tag):
    """(len(1), D) float32, keyed by StudyInstanceUID, cached as npz.

    Cache is keyed by tag (train/test) so the two uids lists never collide.
    dev must be 'cuda' when the cache is missing: the model is ~8B params and
    device_map auto flattens onto CPU when the GPU is absent; on CPU the
    extractor text features from v1 remain the report branch instead.
    """
    fp = Path(f"text_cache_{tag}.npz")
    try:
        z = np.load(fp)
        if list(z["uid"]) == list(uids):
            log(f"text[{tag}]: cache hit {fp.name} ({z['emb'].shape})")
            return z["emb"].astype(np.float32)
    except Exception:
        pass
    if dev.type != "cuda":
        log(f"text[{tag}]: GPU required for Gemma - skipped (T0 carries text)")
        return None
    tokenizer, model = load_text_model()
    if tokenizer is None:
        return None
    log(f"text[{tag}]: embedding {len(uids)} studies")
    D = None
    embs = np.zeros((len(uids), 3840), np.float32)
    B = 8
    for b in range(0, len(uids), B):
        texts = [str(reports.get(u, "")) for u in uids[b:b + B]]
        e = text_embed_batch(model, tokenizer, texts, dev)
        if D is None:
            D = e.shape[1]
            embs = np.zeros((len(uids), D), np.float32)
        embs[b:b + B, :D] = e
    np.savez(fp, uid=np.array(uids, dtype=object), emb=embs.astype(np.float16))
    log(f"text[{tag}]: cached {embs.shape} -> {fp}")
    return embs