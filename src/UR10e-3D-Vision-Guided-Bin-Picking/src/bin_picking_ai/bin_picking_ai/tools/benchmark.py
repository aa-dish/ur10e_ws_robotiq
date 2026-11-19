"""Run "pick <object> and put it in the bin" for every object and print a success table.

  offline (no ROS):  python3 -m bin_picking_ai.tools.benchmark
  with the simulator: start the Gazebo launch, then   ros2 run bin_picking_ai chat
                      and type the same commands (success is reported by the node, using Gazebo ground truth).
"""
import sys
import time

import numpy as np

from .offline_demo import make_system


def main():
    sysm, robot = make_system(use_llm='--llm' in sys.argv, points_per_m2=4e5)
    cat = sysm.catalog
    rows = []
    for cid, e in cat.entries.items():
        for k, ent in enumerate(e.entities):
            n_before = sum(np.all(np.abs(p[:2] - robot.bin_center) < robot.bin_half) for p in robot.entity_poses().values())
            t0 = time.time()
            robot.time = 0.0
            msg = sysm.handle(f'pick the {e.names[0]} and put it in the bin')
            n_after = sum(np.all(np.abs(p[:2] - robot.bin_center) < robot.bin_half) for p in robot.entity_poses().values())
            rows.append((cid, ent, n_after == n_before + 1, robot.time, time.time() - t0))
    print('\n=== benchmark ===')
    for cid, ent, ok, st, wt in rows:
        print(f'{cid:16s} {ent:28s} {"SUCCESS" if ok else "FAIL   "}  sim {st:5.1f}s')
    ok = sum(r[2] for r in rows)
    print(f'success rate: {ok}/{len(rows)} = {100 * ok / len(rows):.0f}%   mean cycle {np.mean([r[3] for r in rows]):.1f}s')
    return 0 if ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
