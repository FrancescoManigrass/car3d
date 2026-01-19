## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run
python run_loop.py \
  --blender /usr/bin/blender \
  --mesh path/to/model_uv.obj \
  --prompt "aggressive sporty automotive design, thinner headlights, sharp character lines, realistic materials" \
  --control_type normal \
  --num_candidates 12 \
  --iters_loop 3 \
  --workdir outputs/run1

## Output
outputs/run1/
  renders/<view>/{rgb_0001.png, depth_0001.exr, depth.png, normal_0001.png, sil_0001.png, edges.png, hardpoints.png}
  candidates/loop_00/<view>/cand_*.png + ranking.json
  chosen/loop_00/<view>.png
  textures/loop_00/optimized_texture.png
  history.json
