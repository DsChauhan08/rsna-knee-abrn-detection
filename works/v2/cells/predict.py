"""predict cell for v2: v1 predict (now text-aware) + submission writers."""

def predict(model, cache, mask, idx, dev, txt=None, feats=False):
    """Logits (n,12), averaging the cached groups.

    feats=True: `cache` carries precomputed (n, N_SLOT, Dv) features
    (feature-cache fast path); otherwise it is the uint8 image cache.
    """

    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), EVAL_BATCH):
            sel = idx[b:b + EVAL_BATCH]
            m = torch.from_numpy(mask[sel]).to(dev)
            t = (torch.from_numpy(txt[sel]).float().to(dev)
                 if txt is not None else None)
            if feats:
                v = torch.from_numpy(cache[sel]).float().to(dev)
                z = model(None, m, t, feats=v)
            else:
                rows = torch.from_numpy(cache[sel]).to(dev)
                imgs = rows[:, :, :GROUP]
                z = model(imgs, m, t)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score

    s = []
    for j in range(p.shape[1]):
        if len(np.unique(y[:, j])) == 2 and np.sum(y[:, j] == 1) > 0:
            try:
                s.append(roc_auc_score(y[:, j], p[:, j]))
            except ValueError:
                pass
    return float(np.mean(s)) if s else float("nan")


def write_benchmark_submission():
    t = pd.read_csv(ROOT / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)


def write_submission(rank_sum, n_models, st_te, test_df):
    P = rank_sum / max(n_models, 1)
    sub = pd.DataFrame(P, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", st_te)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv("submission.csv", index=False)
    return sub