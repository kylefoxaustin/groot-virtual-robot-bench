# source drone/env.sh   -- paths for the drone benchmark (drone/), after `source setup/env.sh`. Override any before sourcing.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
[ -n "${GVRB_WORK:-}" ] || source "$_here/setup/env.sh"
export DRONE_WORK="${DRONE_WORK:-$GVRB_WORK/drone}"            # engines, stand-ins, run outputs
export DRONE_ENG="${DRONE_ENG:-$DRONE_WORK/engines}"           # vit_fp16.engine + llm_fp16.engine (296 tokens) + vlln289/
export DRONE_ASSETS="${DRONE_ASSETS:-$GVRB_ROOT/drone/assets}" # drone_head.npz + gate_set.npz (shipped in this repo)
export DRONE_STANDINS="${DRONE_STANDINS:-$DRONE_WORK/standins}" # seeded contention stand-ins (drone/contention/make_standins.py)
export DRONE_RUNS="${DRONE_RUNS:-$DRONE_WORK/runs}"
export NO_ALBUMENTATIONS_UPDATE=1
unset _here
