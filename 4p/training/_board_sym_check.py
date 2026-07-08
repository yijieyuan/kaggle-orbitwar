import numpy as np, os
z = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "shared", "board_pool_4p", "v2", "boards.npz"))
print("fields:", list(z.files))
po = z["p_owner"]; px = z["p_x"]; py = z["p_y"]; ps = z["p_ships"]
pm = z["p_mask"] if "p_mask" in z.files else np.ones_like(po, bool)
prodf = [f for f in z.files if "prod" in f.lower()]
print("prod fields:", prodf)
pr = z[prodf[0]] if prodf else None
iscom = z["p_is_comet"] if "p_is_comet" in z.files else np.zeros(po.shape, bool)
B = po.shape[0]
print("boards=%d planets/board=%d" % (B, po.shape[1]))
print("")
print("seat | #home | homeShips | homeProd | nNeu<30 | prodNeu<30 | distSun")
for o in range(4):
    homemask = (po == o) & pm & (~iscom)
    nhome = homemask.sum(1)
    hship = (ps * homemask).sum(1) / np.clip(nhome, 1, None)
    hprod = ((pr * homemask).sum(1) / np.clip(nhome, 1, None)) if pr is not None else np.zeros(B)
    hx = (px * homemask).sum(1) / np.clip(nhome, 1, None)
    hy = (py * homemask).sum(1) / np.clip(nhome, 1, None)
    neu = (po == -1) & pm & (~iscom)
    cnt = []; sprod = []
    for b in range(0, B, 8):
        d = np.sqrt((px[b] - hx[b]) ** 2 + (py[b] - hy[b]) ** 2)
        near = neu[b] & (d < 30)
        cnt.append(int(near.sum())); sprod.append(float((pr[b] * near).sum()) if pr is not None else 0.0)
    dsun = np.sqrt((hx - 50) ** 2 + (hy - 50) ** 2)
    print("  %d  | %.2f | %7.1f | %6.2f | %6.2f | %8.2f | %.2f" %
          (o, nhome.mean(), hship.mean(), hprod.mean(), np.mean(cnt), np.mean(sprod), dsun.mean()))
# also: mean home (x,y) per seat to see the fixed quadrants
print("")
print("seat | meanHomeX | meanHomeY (fixed quadrant?)")
for o in range(4):
    homemask = (po == o) & pm & (~iscom)
    nhome = homemask.sum(1)
    hx = (px * homemask).sum(1) / np.clip(nhome, 1, None)
    hy = (py * homemask).sum(1) / np.clip(nhome, 1, None)
    print("  %d  | %.2f | %.2f" % (o, hx.mean(), hy.mean()))
