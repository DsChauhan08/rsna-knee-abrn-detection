"""augment cell for v2: v1 pixel-space aug + rank loss kept verbatim."""

def augment(imgs):

    """Small affine + intensity jitter over the whole bag. No flips."""

    B, S, C, H, W = imgs.shape

    x = imgs.float()

    ang = math.radians((np.random.rand() - 0.5) * 2 * AUG_ROT_DEG)

    sc = 1.0 + (np.random.rand() - 0.5) * 2 * AUG_SCALE

    tx = (np.random.rand() - 0.5) * 2 * AUG_SHIFT

    ty = (np.random.rand() - 0.5) * 2 * AUG_SHIFT

    cos, sin = math.cos(ang) / sc, math.sin(ang) / sc

    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)

    theta = theta.unsqueeze(0).repeat(B * S, 1, 1)

    flat = x.reshape(B * S, C, H, W)

    grid = F.affine_grid(theta, flat.shape, align_corners=False)

    x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",

                      align_corners=False).reshape(B, S, C, H, W)

    scale = 1.0 + (np.random.rand() - 0.5) * 2 * AUG_INTENSITY

    x = (x * scale).clamp(0, 255)

    if USE_VFLIP and np.random.rand() < 0.5:

        x = torch.flip(x, dims=[-2])

    return x


def rank_loss(logits, y, w):

    parts = []

    usable = w > 0

    for j in range(logits.shape[1]):

        pos = logits[(y[:, j] > RANK_POS) & usable[:, j], j]

        neg = logits[(y[:, j] < RANK_NEG) & usable[:, j], j]

        if len(pos) and len(neg):

            parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())

    return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)