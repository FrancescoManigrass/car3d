"""Blender render script placeholder.

Usage:
  blender -b scene.blend -P blender_render.py -- '{"output_dir":"/path","depth_near":0.1,"depth_far":100.0}'

Populate this script with Blender API calls to render RGB, depth, normals, silhouettes,
edges, and hardpoints heatmaps.
"""

import json
import sys
from pathlib import Path


def parse_args() -> dict:
    if "--" not in sys.argv:
        return {}
    payload = sys.argv[sys.argv.index("--") + 1 :]
    if not payload:
        return {}
    return json.loads(payload[0])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.get("output_dir", "./renders"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Blender render stub. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
