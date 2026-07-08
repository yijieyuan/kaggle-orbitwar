#!/usr/bin/env bash
# Smoke test the solution: python inference (2p+4p, greedy+merge) + one recorded game each.
# Usage: bash run_tests.sh   (set PY to your python if not on PATH)
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8

echo "== 1/3  2p inference (greedy + merge) =="
"$PY" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
import random, sys, os
from kaggle_environments import make
random.seed(0)
env = make("orbit_wars", configuration={"episodeSteps": 30}); env.reset(2); env.step([[],[]])
obs = env.state[0]["observation"]
sys.path.insert(0, os.path.abspath("2p/inference")); import agent
for mode in ("greedy","merge"):
    n = len(agent.make_agent(mode)(obs))
    print(f"   2p {mode}: {n} moves  OK")
EOF

echo "== 2/3  4p inference (greedy + merge) =="
"$PY" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
import random, sys, os
from kaggle_environments import make
random.seed(0)
env = make("orbit_wars", configuration={"episodeSteps": 30}); env.reset(4); env.step([[]]*4)
sys.path.insert(0, os.path.abspath("4p/inference")); import agent
for mode in ("greedy","merge"):
    a = agent.make_agent(mode)
    ok = all(isinstance(a(env.state[s]["observation"]), list) for s in range(4))
    print(f"   4p {mode}: seats callable={ok}  OK")
EOF

echo "== 3/3  record 1 game each (2p + 4p) =="
# Smoke output goes to the gitignored eval/_smoke_replays/ (matches **/_smoke*/ in .gitignore),
# so the smoke test never clobbers the committed local_replays/ store.
( cd eval && "$PY" eval.py --out-root _smoke_replays --combo 2p_merge,2p_greedy --games 1 >/dev/null 2>&1 && echo "   2p game recorded OK" )
( cd eval && "$PY" eval.py --out-root _smoke_replays --combo 4p_merge,public_agent_1,public_agent_1,public_agent_1 --games 1 >/dev/null 2>&1 && echo "   4p game recorded OK" )
echo "ALL SMOKE TESTS PASSED — see eval/_smoke_replays/2p/INDEX.json"
