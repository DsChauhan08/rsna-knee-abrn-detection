"""main cell for v2 (ceiling path): clean supervision, CV on confident
labels, and optional vision fine-tuning during the fold loop."""

# ---- orchestrator --------------------------------------------------------

REQUIRE_GPU = True

# --- ceiling levers -------------------------------------------------------
TGT_CONF_FLOOR = 0.45   # drop lexicon pseudo-labels below this conf (noise)
CLEAN_CONF = 0.50       # validation AUC only on targets at/above conf
FINE_TUNE = True        # unfreeze the last N vision blocks during folds
FINE_TUNE_BLOCKS = 6
FT_LR = 3e-5            # encoder-block LR (head keeps LR_HEAD)
FT_EPOCHS = 2           # fine-tune epochs per fold (pixel path)
EPOCHS_FAST = 4         # frozen-feature-path epochs per fold


def tune_threads():
    """Scale the v1 thread pools to this instance's vCPUs. The DICOM->cache
    phase is the run's floor (thread-bound), so oversubscription of the
    actual core count is the only lever that moves it."""
    try:
        n = os.cpu_count() or 8
        globals()["HDR_THREADS"] = max(HDR_THREADS, min(n, 64))
        globals()["PIX_THREADS"] = max(PIX_THREADS, min(n, 48))
        globals()["ORDER_THREADS"] = max(ORDER_THREADS, min(n * 2, 96))
        os.environ["OMP_NUM_THREADS"] = str(min(n, 16))
        os.environ["OPENBLAS_NUM_THREADS"] = str(min(n, 16))
        log(f"threads: hdr {HDR_THREADS} pix {PIX_THREADS} "
            f"order {ORDER_THREADS} (vCPUs {n})")
    except Exception as e:
        log(f"thread tuning failed: {e!r}")


def build_clean_targets(st_tr, gold, lab, floor=TGT_CONF_FLOOR):
    """Y/W as in v1, plus the per-target confidence `C` (1.0 for gold).

    Lexicon pseudo-labels whose extractor confidence is below the floor get
    weight 0 - they are noise, not supervision; gold labels are left at
    GOLD_WEIGHT. keep = rows carrying at least one supervised target.
    """
    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    C = np.zeros_like(Y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = gold.loc[st].values, GOLD_WEIGHT
            C[i] = 1.0
        elif st in lab.index:
            r = lab.loc[st]
            Y[i] = r[TARGETS].values
            C[i] = r[[t + "__conf" for t in TARGETS]].values
            W[i] = 0.25 + 0.75 * C[i]
            W[i][C[i] < floor] = 0.0
    return Y, W, W.sum(1) > 0, C


def clean_auc(y, p, conf, floor):
    """macro-AUC over the confident subset: only (sample,target) pairs
    whose extractor confidence >= floor and both classes are present.
    Targets are binarised like v1 (pseudo-labels are graded scores)."""
    from sklearn.metrics import roc_auc_score
    yb = (y > 0.5).astype(int)
    s, n = [], 0
    for j in range(p.shape[1]):
        m = conf[:, j] >= floor
        if m.sum() < 16 or len(np.unique(yb[m, j])) < 2:
            continue
        try:
            s.append(roc_auc_score(yb[m, j], p[m, j]))
            n += 1
        except ValueError:
            continue
    return (float(np.mean(s)) if s else float("nan")), n


def main():
    tune_threads()                             # before the DICOM phase
    write_benchmark_submission()             # scoreable from second 1

    def pick_device():
        # CUDA if a real kernel is runnable, else CPU.
        # Recent torch builds dropped Pascal (P100, sm_60) kernels, so
        # cuda.is_available() can be False even on GPU machines; and on
        # older builds the first forward can die with "no kernel image is
        # available". A probe op + explicit logging catches both.
        import torch as _t
        if not _t.cuda.is_available():
            log(f"pick_device: no CUDA build "
                f"(torch {_t.__version__}) -> cpu")
            return "cpu"
        try:
            _t.ones(2, device="cuda").float().sum().item()
            _t.cuda.synchronize()
        except Exception as e:
            log(f"pick_device: probe failed ({type(e).__name__}: "
                f"{str(e)[:80]}) -> cpu")
            return "cpu"
        log(f"pick_device: cuda ({_t.cuda.get_device_name(0)})")
        return "cuda"

    dev = torch.device(pick_device())
    if REQUIRE_GPU and dev.type != "cuda":
        log("GPU required but not runnable; keeping the 0.5 benchmark "
            "submission and stopping now (no DICOM burn, no CPU burn). "
            "Pick a T4 GPU on Kaggle, or an older container image "
            "(torch with Pascal kernels) on a P100.")
        return
    log(f"compute device: {dev}"
        f"{' (' + torch.cuda.get_device_name(0) + ')' if dev.type == 'cuda' else ''}")

    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    train_df = pd.read_csv(ROOT / "train.csv")
    train_series = pd.read_csv(ROOT / "train_series.csv")
    log(f"train {train_df.shape} test {test_df.shape}")

    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    htr = annotate(walk("train_series"))
    hte = annotate(walk("test_series"))
    lat_tr, _ = laterality_maps(htr)
    lat_te, _ = laterality_maps(hte)
    slots_tr = pick_slots(htr, plane_map)
    slots_te = pick_slots(hte, plane_map)

    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_tr, "train")
    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_te, "test")

    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    lab = read_labels(train_df)
    Y, W, keep, Conf = build_clean_targets(st_tr, gold, lab, TGT_CONF_FLOOR)
    log(f"supervised {int(keep.sum())} of {len(st_tr)} studies "
        f"(conf floor {TGT_CONF_FLOOR})")

    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(str(rep.get(s, s)).encode()).hexdigest()[:8], 16)
                    % N_FOLDS for s in st_tr])
    gpos = {s: i for i, s in enumerate(st_tr)}
    gi_all = np.array([gpos[s] for s in gold.index if s in gpos])

    vision = load_vision(dev)
    use_ft = (FINE_TUNE and isinstance(vision, HFEncoder)
              and dev.type == "cuda")
    if use_ft:
        nft = vision.set_grad(FINE_TUNE_BLOCKS)
        log(f"fine-tune ON: last {FINE_TUNE_BLOCKS} blocks trainable "
            f"({nft:,} params), {FT_EPOCHS} epochs/fold")
    else:
        log(f"fine-tune OFF: frozen-encoder feats path "
            f"(encoder {type(vision).__name__})")

    te_tr = build_text_cache(st_tr, rep.to_dict(), dev, "train")
    try:
        rep_te = test_df.set_index("StudyInstanceUID")["Report"].fillna("")
        te_te = build_text_cache(st_te, rep_te.to_dict(), dev, "test")
    except Exception:
        te_te = None
    text_dim = (te_tr.shape[1] if te_tr is not None else 0)
    if text_dim:
        log(f"text: {text_dim}-d report embeddings (train {te_tr is not None}, "
            f"test {te_te is not None})")
    else:
        log("text: no Gemma embeddings - extractor T0 is the text branch")

    feats_tr = feats_te = None
    if use_ft:
        # fine-tuning needs the pixel path: no feature cache, keep the
        # image caches alive for the fold loop
        _EPOCHS = FT_EPOCHS
    else:
        feats_tr = build_feature_cache(vision, Ctr, st_tr, "train", dev)
        feats_te = build_feature_cache(vision, Cte, st_te, "test", dev) \
            if feats_tr is not None else None
        if feats_tr is not None and feats_te is not None:
            del Ctr, Cte
            gc.collect()
            log("feature cache active - image caches freed")
            from copy import deepcopy
            vision = deepcopy(FeatBackbone(vision.feature_dim, dev))
            _EPOCHS = EPOCHS_FAST
        else:
            _EPOCHS = EPOCHS
    log(f"encoder {type(vision).__name__} dim={vision.feature_dim} "
        f"epochs/fold={_EPOCHS}")

    rank_sum = np.zeros((len(st_te), len(TARGETS)), np.float64)
    n_models, sub = 0, None

    for fold in range(N_FOLDS):
        va = np.array([i for i in range(len(st_tr))
                       if keep[i] and grp[i] == fold])
        tr = np.array([i for i in range(len(st_tr))
                       if keep[i] and grp[i] != fold])
        if len(va) == 0 or len(tr) < 2:
            log(f"fold {fold}: too small, skipped"); continue
        gi_va = np.array([i for i in gi_all if grp[i] == fold])
        log(f"=== fold {fold}: train {len(tr)} holdout {len(va)}"
            f" (gold held out {len(gi_va)}) ===")
        t0 = time.time()
        model = build_v2_model(vision, text_dim).to(dev)
        # two LR groups in fine-tune mode: encoder (slow) vs head (fast)
        body = [p for n_, p in model.named_parameters()
                if p.requires_grad and n_.startswith("vision.")]
        head = [p for p in model.parameters() if p.requires_grad
                and all(p is not q for q in body)]
        groups = [{"params": head, "lr": LR_HEAD}]
        if use_ft and body:
            groups = [{"params": body, "lr": FT_LR}] + groups
        opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        ema = Ema(model, EMA_DECAY,
                  skip_prefixes=("vision.",) if use_ft else ())
        steps = max(_EPOCHS * max(len(tr) // BATCH_STUDIES, 1), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups],
            total_steps=steps, pct_start=0.15)
        yv = (Y[va] > 0.5).astype(int)
        best, best_state = -1.0, None

        for ep in range(_EPOCHS):
            model.train()
            perm = np.random.permutation(tr)
            tot = nst = 0
            for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
                sel = perm[b:b + BATCH_STUDIES]
                m_ = torch.from_numpy(Mtr[sel]).to(dev)
                y = torch.from_numpy(Y[sel]).to(dev)
                w_ = torch.from_numpy(W[sel]).to(dev)
                t_ = (torch.from_numpy(te_tr[sel]).float().to(dev)
                      if te_tr is not None else None)
                if feats_tr is not None:
                    feats = torch.from_numpy(feats_tr[sel]).to(dev)
                    feats = feats + torch.randn_like(feats) * 0.03
                    z = model(None, m_, t_, feats=feats)
                else:
                    rows = torch.from_numpy(Ctr[sel]).to(dev)
                    imgs = augment(rows[:, :, :GROUP])
                    z = model(imgs, m_, t_)
                loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none")
                        * w_).mean()
                if RANK_LOSS_W > 0:
                    loss = loss + RANK_LOSS_W * rank_loss(z.float(), y, w_)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                ema.update(model)
                tot += float(loss); nst += 1
            em = build_eval_model(vision, text_dim, dev, src=model)
            ema.target(em)
            if feats_tr is not None:
                pv = predict(em, feats_tr, Mtr, va, dev, te_tr, feats=True)
            else:
                pv = predict(em, Ctr, Mtr, va, dev, te_tr)
            d, ns = clean_auc(yv, pv, Conf, CLEAN_CONF)
            log(f"fold {fold} ep {ep + 1}/{_EPOCHS} "
                f"loss {tot / max(nst, 1):.4f} clean {d:.4f} ({ns} targets)")
            if d > best:
                best = d
                best_state = {k: v.detach().cpu().clone()
                              for k, v in em.state_dict().items()}
            if time.time() - T0 > TIME_BUDGET:
                log("time budget reached"); break

        if best_state is None:
            em = build_eval_model(vision, text_dim, dev, src=model)
            ema.target(em)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in em.state_dict().items()}
        model = build_v2_model(vision, text_dim).to(dev)
        model.load_state_dict(best_state, strict=False)
        if feats_te is not None:
            P = predict(model, feats_te, Mte, np.arange(len(st_te)), dev,
                        te_te, feats=True)
        else:
            P = predict(model, Cte, Mte, np.arange(len(st_te)), dev, te_te)
        rank_sum += pd.DataFrame(P).rank(pct=True).values
        n_models += 1
        sub = write_submission(rank_sum, n_models, st_te, test_df)
        log(f"fold {fold} done in {time.time() - t0:.0f}s")

    if sub is None:
        log("no fold finished; the 0.5 benchmark submission stands")
        return
    log(f"submission.csv {sub.shape}; models {n_models}")
    print(sub.head().to_string())


try:
    main()
except Exception:
    traceback.print_exc()
    log("run failed; the 0.5 benchmark submission is on disk")