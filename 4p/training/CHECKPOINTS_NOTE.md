# 4p training checkpoints

The 4p model is `experiment-025-simplefrac` (2-tree {net,frac}, native 4-seat, self-play PPO).

- **`checkpoints/`** -- the FULL deploy training run: 47 checkpoints `ckpt_u01000..u47000.msgpack`
  (every 1000 updates). Verified byte-exact: `ckpt_u44000` == the deployed weights
  `../inference/weights/weights_4p_u44000.npz` (142/142 arrays match), and `ckpt_u39000` == the merge
  partner. This is the exact run behind Kaggle submissions 53993524 / 53993338.
- **`checkpoints_il/`** -- imitation-learning warm-start trajectory (`weights_il__u06000..u49999.npz`,
  13 snapshots) + `train_il.log`, the pre-train the RL fine-tune started from.

(The 2p track under `../../2p/training/` has its full deploy run: 166 model ckpts u1000–u55200 (+132
optimizer-state files = 298 files); the deployed weight u55000 is byte-verified.)
