"""4p inference entrypoint — final submission model (experiment-025-simplefrac, 4p).

make_agent(mode) returns a callable agent(obs, config=None) -> list[move].

  mode="greedy"  -> single checkpoint u44000, greedy decode (no rollout).
                    == Kaggle submission 53993524 (4p side).
  mode="merge"   -> value-ensemble MERGE of u44000 (A/main) + u39000 (B), adaptive-H
                    rollout when the agents disagree. == Kaggle submission 53993338 (4p side).

The 4p stack is FLAT in this dir (engine.py, rl_infer.py, agent_greedy/merge.py, forward_sim.py,
sim_runner.py, utils/). Its rl_infer/engine SHARE names with the 2p ones, so a single process must
NOT import both tracks (eval.py runs each in its own subprocess). Pure-numpy (no torch). Weights in ./weights/.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_W = os.path.join(_HERE, "weights")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def make_agent(mode="merge"):
    if mode == "greedy":
        os.environ["ORBIT4P_WEIGHTS"] = os.path.join(_W, "weights_4p_u44000.npz")
        from agent_greedy import agent
        return agent
    if mode == "merge":
        os.environ["ORBIT4P_MERGE_A"] = os.path.join(_W, "weights_4p_u44000.npz")
        os.environ["ORBIT4P_MERGE_B"] = os.path.join(_W, "weights_4p_u39000.npz")
        from agent_merge import agent
        return agent
    raise ValueError(f"mode must be 'greedy' or 'merge', got {mode!r}")


agent = make_agent("merge")
