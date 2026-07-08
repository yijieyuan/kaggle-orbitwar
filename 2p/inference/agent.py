"""2p inference entrypoint — final submission model (experiment-026-sample).

make_agent(mode) returns a callable agent(obs, config=None) -> list[move].

  mode="greedy"  -> single checkpoint u55000, reach-masked greedy decode (no rollout).
                    == Kaggle submission 53993524 (2p side).
  mode="merge"   -> value-ensemble MERGE of u55000 (A/main) + u53000 (B), adaptive-H
                    rollout arbitration when the agents disagree. == Kaggle submission 53993338 (2p side).

Both are pure-numpy (no torch). Weights live in ./weights/. The two rl_agent_*.py modules
load their weights at import time, so we set the weight-path env vars BEFORE importing.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_W = os.path.join(_HERE, "weights")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def make_agent(mode="merge"):
    if mode == "greedy":
        os.environ["ORBIT25_2P_WEIGHTS"] = os.path.join(_W, "weights_2p_u55000.npz")
        from rl_agent_greedy import agent
        return agent
    if mode == "merge":
        os.environ["ORBIT25_2P_WEIGHTS_A"] = os.path.join(_W, "weights_2p_u55000.npz")
        os.environ["ORBIT25_2P_WEIGHTS_B"] = os.path.join(_W, "weights_2p_u53000.npz")
        from rl_agent_merge import agent
        return agent
    raise ValueError(f"mode must be 'greedy' or 'merge', got {mode!r}")


# Default entrypoint = merge (the higher-scoring of our two final subs).
agent = make_agent("merge")
