"""Export exp22 2-tree flax msgpack {net, frac} -> flat .npz for rl_infer / agent.py.
  net  subtree -> flat names (Conv_0/kernel, Dense_3/bias, LayerNorm_2/scale ...)
  frac subtree -> 'frac/' prefixed (CoordFracGauss).
Run: python export_weights.py checkpoints/ckpt_uXXXXX.msgpack weights.npz   (weights.npz = agent.py's default)
"""
import sys
import numpy as np
import flax.serialization as fser

src = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_u08000.msgpack"
dst = sys.argv[2] if len(sys.argv) > 2 else "weights.npz"
tree = fser.msgpack_restore(open(src, "rb").read())
assert all(k in tree for k in ("net", "frac")), f"not a v5 2-tree {{net,frac}} ckpt: {list(tree)}"

flat = {}
def walk(node, prefix):
    for k, v in node.items():
        name = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            walk(v, name)
        else:
            flat[name] = np.asarray(v)

for tname, pfx in (("net", ""), ("frac", "frac")):
    sub = tree[tname]
    sub = sub.get("params", sub)
    walk(sub, pfx)
np.savez(dst, **flat)
print(f"wrote {dst}: {len(flat)} arrays "
      f"(net {sum(1 for k in flat if not k.startswith('frac/'))}, "
      f"frac {sum(1 for k in flat if k.startswith('frac/'))})")
