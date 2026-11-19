"""The brain: text command -> detect -> choose object -> plan grasp -> plan motion -> execute -> verify.

No ROS imports: talks to the robot through :class:`RobotInterface`.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .geometry import wrap_half_pi
from .grasping import GraspPlanner, GripperModel
from .kinematics import UR10e
from .llm import Command, CommandParser, resolve_targets
from .motion import PickPlacePlanner, PlanError
from .motion.plan import BlockedError
from .robot_interface import RobotInterface
from .vision import Catalog, Detector, VisionParams
from .vision.detector import Detection, DetectionResult


def selected_gripper() -> str:
    """'2f140' (default) or '2f85' - chosen with the environment variable UR_GRIPPER (also read by the Gazebo launch file)."""
    g = os.environ.get('UR_GRIPPER', '2f140').strip().lower()
    if g not in ('2f140', '2f85'):
        raise ValueError(f"UR_GRIPPER must be '2f140' or '2f85', got '{g}'")
    return g


def gripper_config_file(config_dir: str) -> str:
    return os.path.join(config_dir, f'gripper_{selected_gripper()}.yaml')


class NoGraspError(PlanError):
    """Object cannot be grasped by this gripper at all (too wide / too small)."""


@dataclass
class SystemConfig:
    catalog_path: str
    places_path: str
    gripper_path: str
    world_objects_path: str = ''
    left_right: str = 'image'            # 'image' (left = smaller x as seen in the camera picture) | 'robot'
    max_retries: int = 2
    verify: bool = True
    robot_base_world: Tuple[float, float, float] = (0.842175, -0.500138, 0.6)


class BinPickingSystem:
    skipped_ungraspable = False

    def __init__(self, robot: RobotInterface, cfg: SystemConfig, parser: Optional[CommandParser] = None,
                 vision_params: Optional[VisionParams] = None):
        self.robot = robot
        self.cfg = cfg
        self.catalog = Catalog(cfg.catalog_path)
        self.places = yaml.safe_load(open(cfg.places_path, encoding='utf-8'))
        gcfg = yaml.safe_load(open(cfg.gripper_path, encoding='utf-8'))['gripper']
        self.gripper = GripperModel.from_dict(gcfg)
        self.robot_model = UR10e(self.gripper.tcp_z)
        self.grasp_planner = GraspPlanner(self.gripper)
        self.detector = Detector(self.catalog, vision_params)
        self.parser = parser or CommandParser(self.catalog, None, use_llm=False)
        self.home_q = np.array(self.places['home_q'], float)
        self._abort = threading.Event()
        self._busy = threading.Lock()
        self.last_result: Optional[DetectionResult] = None
        self.placed_entities: List[str] = []
        self._initial_poses: Dict[str, np.ndarray] = {}
        if cfg.world_objects_path and os.path.exists(cfg.world_objects_path):
            base = np.array(cfg.robot_base_world)
            for o in json.load(open(cfg.world_objects_path)):
                self._initial_poses[o['name']] = np.array(o['pose'][:3]) - base
        self.log_lines: List[str] = []

    # ------------------------------------------------------------------ utilities
    def say(self, msg: str):
        self.log_lines.append(msg)
        self.robot.say(msg)

    def abort(self):
        self._abort.set()

    def _aborted(self) -> bool:
        return self._abort.is_set()

    def at_home(self, tol: float = 0.08) -> bool:
        return bool(np.max(np.abs(self.robot.get_arm_q() - self.home_q)) < tol)

    def go_home(self) -> bool:
        q = self.robot.get_arm_q()
        if self.at_home():
            return True
        pp = self._planner(0.127)
        st = pp.joint_segment(q, self.home_q, 'to_home', speed=0.8)
        return self.robot.execute_arm(st, self._aborted)

    def _planner(self, table_z: float) -> PickPlacePlanner:
        return PickPlacePlanner(self.robot_model, self.gripper, self.places, table_z)

    # ------------------------------------------------------------------ perception
    def observe(self, go_home: bool = True) -> DetectionResult:
        if go_home and not self.at_home():
            self.say('Moving the arm out of the camera view ...')
            self.go_home()
            self.robot.sleep(0.8)
        rgb, depth, K, T = self.robot.capture()
        res = self.detector.detect(rgb, depth, K, T)
        self.last_result = res
        return res

    def describe_scene(self, res: DetectionResult) -> str:
        if not res.detections:
            return 'I do not see any objects on the table.'
        lines = ['I see %d objects:' % len(res.detections)]
        for d in sorted(res.detections, key=lambda d: d.center[0]):
            nm = self.catalog.display_name(d.label) if d.label else 'unknown object'
            lines.append(f'  #{d.id} {nm} ({d.color_name}, {d.height * 100:.0f} cm tall) at '
                         f'x={d.center[0]:+.2f} y={d.center[1]:+.2f}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ commands
    def handle(self, text: str) -> str:
        """Run one user instruction (blocking).  Returns the final message."""
        low = text.strip().lower()
        if low in ('stop', 'halt', 'abort', 'cancel'):
            self.abort()
            self.say('Stopping.')
            return 'stopped'
        if not self._busy.acquire(blocking=False):
            msg = 'I am still busy with the previous command. Say "stop" to interrupt.'
            self.say(msg)
            return msg
        try:
            self._abort.clear()
            res = self.observe()
            cmd = self.parser.parse(text, res.detections)
            return self._dispatch(cmd, res)
        except Exception as e:                                         # never crash the node
            msg = f'Something went wrong: {type(e).__name__}: {e}'
            self.say(msg)
            return msg
        finally:
            self._busy.release()

    def _dispatch(self, cmd: Command, res: DetectionResult) -> str:
        if cmd.action == 'help':
            msg = ("I can pick objects from the table and put them in the bin or somewhere else. Examples: "
                   "'pick the mustard bottle and put it in the bin', 'move the pink box to the right', "
                   "'clear the table', 'what do you see', 'go home', 'stop'.")
            self.say(msg)
            return msg
        if cmd.action == 'home':
            ok = self.go_home()
            msg = 'I am at the home position.' if ok else 'Could not go home.'
            self.say(msg)
            return msg
        if cmd.action == 'list':
            msg = self.describe_scene(res)
            self.say(msg)
            return msg
        if cmd.action == 'status':
            msg = f'Idle. {len(res.detections)} objects visible, {len(self.placed_entities)} moved so far.'
            self.say(msg)
            return msg
        if cmd.action != 'pick_place':
            msg = cmd.reply or "Sorry, I didn't understand that."
            self.say(msg)
            return msg
        targets = resolve_targets(cmd, res.detections, self.catalog, self.cfg.left_right)
        if not targets:
            msg = f'I could not find "{cmd.object_phrase or cmd.raw}" on the table.\n' + self.describe_scene(res)
            self.say(msg)
            return msg
        self.say(f'Understood ({cmd.source}): pick {", ".join(self._name(d) for d in targets[:5])}'
                 f'{"..." if len(targets) > 5 else ""} -> {cmd.target}')
        results = []
        pending = [(d.label, d.center[:2]) for d in targets]
        failures = 0
        for label, xy in pending:
            if self._aborted():
                break
            self.skipped_ungraspable = False
            ok = self.pick_place(label, xy, cmd.target)
            results.append(ok)
            failures = 0 if (ok or self.skipped_ungraspable) else failures + 1
            if failures >= 2:
                self.say('Two failures in a row - stopping.')
                break
        self.go_home()
        done = sum(results)
        msg = f'Done: {done}/{len(pending)} objects moved to {cmd.target}.' if not self._aborted() else 'Stopped by user.'
        self.say(msg)
        return msg

    def _name(self, d: Detection) -> str:
        return self.catalog.display_name(d.label) if d.label else f'object #{d.id}'

    # ------------------------------------------------------------------ one object
    def pick_place(self, label: Optional[str], approx_xy, target: str, _depth: int = 0) -> bool:
        attempt = -1
        clears = 0
        while attempt + 1 < 1 + self.cfg.max_retries:
            attempt += 1
            if self._aborted():
                return False
            res = self.observe()
            det = self._find(res, label, approx_xy)
            if det is None:
                self.say('That object is no longer on the table - maybe it was already moved.')
                return attempt > 0
            name = self._name(det)
            self.say(f'[{attempt + 1}] Picking {name} (height {det.height * 100:.0f} cm, '
                     f'{det.size[0] * 100:.0f}x{det.size[1] * 100:.0f} cm) ...')
            try:
                plan, entity, pinfo = self._make_plan(res, det, target)
            except BlockedError as e:
                if _depth < 2 and clears < 4 and e.blockers:
                    byid = {d.id: d for d in res.detections}
                    blk = None
                    for bid in e.blockers:
                        cand = byid.get(bid)
                        if cand is not None and self.grasp_planner.plan(cand, res):
                            blk = cand
                            break
                    if blk is not None:
                        self.say(f'  {name} is hemmed in by {self._name(blk)} - moving that out of the way first')
                        moved = self.pick_place(blk.label, blk.center[:2], 'table_center', _depth + 1)
                        clears += 1
                        attempt -= 1                      # clearing a neighbour is not a failed attempt
                        if not moved:
                            self.say(f'  could not move {self._name(blk)}')
                            return False
                        continue
                if e.blockers and _depth < 2 and clears < 4:
                    self.say(f'  skipping {name}: hemmed in by a neighbour this gripper cannot pick up (too wide)')
                    self.skipped_ungraspable = True
                    return False
                self.say(f'  cannot plan: {e}')
                continue
            except NoGraspError as e:
                self.say(f'  skipping {name}: {e}')
                self.skipped_ungraspable = True
                return False
            except PlanError as e:
                self.say(f'  cannot plan: {e}')
                continue
            ok = self._execute(plan)
            if self._aborted():
                return False
            if not ok:
                self.say('  execution failed - will retry')
                self.go_home()
                continue
            if self.cfg.verify:
                if self._verify(label, approx_xy, entity, target):
                    self.say(f'  OK {name} placed ({target}).')
                    if target == 'bin':
                        self.placed_entities.append(entity)
                    return True
                self.say('  the object is still on the table - retrying')
                continue
            if target == 'bin':
                self.placed_entities.append(entity)
            return True
        self.say(f'  giving up on {self._name(det) if det else label}.')
        return False

    def _find(self, res: DetectionResult, label: Optional[str], xy) -> Optional[Detection]:
        same = [d for d in res.detections if d.label == label] if label else list(res.detections)
        if not same:
            # The recogniser labels every detection from the objects still visible, so once a look-alike is gone
            # a neighbour can get a different label.  The object itself has not moved: accept whatever stands at that spot.
            near = [d for d in res.detections if math.hypot(d.center[0] - xy[0], d.center[1] - xy[1]) < 0.04]
            if near and label:
                d = min(near, key=lambda d: math.hypot(d.center[0] - xy[0], d.center[1] - xy[1]))
                import dataclasses
                self.say(f'  ({self.catalog.display_name(label)} is now recognised as {self._name(d)}, but it is still the same object)')
                return dataclasses.replace(d, label=label)
            return None
        return min(same, key=lambda d: math.hypot(d.center[0] - xy[0], d.center[1] - xy[1]))

    # ------------------------------------------------------------------ planning
    def _make_plan(self, res: DetectionResult, det: Detection, target: str):
        pp = self._planner(res.table_z)
        mot = self.places['motion']
        maxh = max([d.height for d in res.detections] + [0.1])
        travel_z = res.table_z + min(maxh + mot['travel_margin'], mot['max_travel_above_table'])
        grasps = self.grasp_planner.plan(det, res)
        if not grasps:
            raise NoGraspError('no feasible grasp (object too wide / too small for the gripper)')
        q_now = self.robot.get_arm_q()
        last = None
        free = [g for g in grasps if g.clearance >= -0.004]
        if not free:                                    # every approach is blocked -> tell the caller who is in the way
            best = min(grasps, key=lambda g: (len(g.blockers), sum(g.blockers.values())))
            order = sorted(best.blockers, key=lambda k: -best.blockers[k])
            raise BlockedError(f'open fingers would hit a neighbour ({best.notes})', order)
        for g in free[:8]:
            place = self._place_for(res, det, g, target)
            entity = self._entity_for(det, g.center_xy)
            try:
                plan = pp.plan_pick_place(q_now, g, entity, det.height, place, travel_z, self.home_q)
                plan.info['grasp'] = g
                return plan, entity, place
            except PlanError as e:
                last = e
        raise last or PlanError('no plan')

    def _place_for(self, res: DetectionResult, det: Detection, g, target: str) -> Dict:
        if target == 'bin':
            b = self.places['bin']
            n = len([e for e in self.placed_entities])               # spread objects inside the bin a little
            ox = ((n % 3) - 1) * 0.07
            oy = (((n // 3) % 3) - 1) * 0.10
            ox = max(-b['half_size'][0] + 0.06, min(b['half_size'][0] - 0.06, ox))
            oy = max(-b['half_size'][1] + 0.08, min(b['half_size'][1] - 0.08, oy))
            return dict(kind='bin', xy=(b['center'][0] + ox, b['center'][1] + oy))
        region = self.places['table_places'].get(target)
        if region is None:
            raise PlanError(f'unknown destination "{target}"')
        found = None
        for margin in (0.018, 0.010, 0.004):                 # relax the safety margin if the table is crowded
            found = self._free_spot(res, det, region, keep_inside=self.places['table_bounds'], margin=margin)
            if found is None:                                # region full -> closest free spot anywhere on the table
                x0, x1, y0, y1 = region
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                whole = (cx - 0.9, cx + 0.9, cy - 0.9, cy + 0.9)
                found = self._free_spot(res, det, whole, centre=(cx, cy), keep_inside=self.places['table_bounds'],
                                        margin=margin)
                if found is not None:
                    self.say(f'  "{target}" is crowded - using the closest free spot ({margin * 100:.0f} cm margin).')
            if found is not None:
                break
        if found is None:
            raise PlanError(f'no free space on the table for "{target}" - the table is too crowded')
        xy, rot = found
        return dict(kind='table', xy=xy, yaw=g.closing_yaw + rot, z=g.tcp_z + 0.012)

    @staticmethod
    def _free_spot(res: DetectionResult, det: Detection, region, radius: float = 0.0, centre=None,
                   keep_inside=None, margin: float = 0.018,
                   rotations=(0.0, 0.5236, -0.5236, 1.0472, -1.0472, 1.5708)):
        """Closest free spot (bare table) for the object's footprint.  Returns ((x, y), rotation) or None.

        The footprint is an oriented rectangle; the object may be rotated by one of ``rotations`` [rad]
        (the wrist turns it while carrying), nearest-to-region-centre wins, small rotations are preferred."""
        own = res.label_map == det.id
        bare_table = np.isfinite(res.height_map) & (np.abs(np.nan_to_num(res.height_map, nan=9.0)) < 0.012)
        occupied = ~(bare_table | own)                    # objects, floor, desk, unseen cells
        ny, nx = occupied.shape
        x0, x1, y0, y1 = region
        cx, cy = centre if centre else ((x0 + x1) / 2, (y0 + y1) / 2)
        half_l = det.size[0] / 2 + margin
        half_w = det.size[1] / 2 + margin
        step = max(int(0.01 / res.cell), 1)
        best, best_cost = None, 1e9
        for rot in rotations:
            yaw = det.yaw + rot
            c_, s_ = math.cos(yaw), math.sin(yaw)
            px = max(int(math.hypot(half_l, half_w) / res.cell) + 1, 2)
            kern = np.zeros((2 * px + 1, 2 * px + 1), np.uint8)
            corners = [(px + (sx * half_l * c_ - sy * half_w * s_) / res.cell,
                        px + (sx * half_l * s_ + sy * half_w * c_) / res.cell)
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            cv2.fillPoly(kern, [np.round(np.array(corners)).astype(np.int32)], 1)
            occ = cv2.dilate(occupied.astype(np.uint8), kern)
            ex = abs(c_) * half_l + abs(s_) * half_w
            ey = abs(s_) * half_l + abs(c_) * half_w
            rows = np.arange(px, ny - px, step)
            cols = np.arange(px, nx - px, step)
            R, C = np.meshgrid(rows, cols, indexing='ij')
            free = occ[R, C] == 0
            X = res.origin[0] + (C + 0.5) * res.cell
            Y = res.origin[1] + (R + 0.5) * res.cell
            ok = free & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            if keep_inside:
                bx0, bx1, by0, by1 = keep_inside
                ok &= (X >= bx0 + ex) & (X <= bx1 - ex) & (Y >= by0 + ey) & (Y <= by1 - ey)
            if not ok.any():
                continue
            cost = np.where(ok, (X - cx) ** 2 + (Y - cy) ** 2 + 0.01 * abs(rot), 1e9)
            i = np.unravel_index(int(np.argmin(cost)), cost.shape)
            if cost[i] < best_cost:
                best_cost, best = float(cost[i]), ((float(X[i]), float(Y[i])), rot)
        return best

    def _entity_for(self, det: Detection, grasp_xy) -> str:
        """Simulator entity name that is under the gripper (needed by the virtual grasp)."""
        cands = list(self.catalog.entries[det.label].entities) if det.label else []
        poses = self.robot.entity_poses()
        if not poses:
            poses = {k: v for k, v in self._initial_poses.items() if k not in self.placed_entities}
        pool = [c for c in cands if c in poses] or [c for c in poses]
        if not pool:
            return cands[0] if cands else ''
        gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
        return min(pool, key=lambda c: math.hypot(poses[c][0] - gx, poses[c][1] - gy))

    # ------------------------------------------------------------------ execution
    def _execute(self, plan) -> bool:
        held = None                                   # entity currently welded to the gripper
        ok = self._execute_steps(plan, lambda e: None)
        return ok

    def _release_again(self, entity: str) -> None:
        """Detach + open the fingers once more (used when the object is still travelling with the gripper)."""
        for _ in range(5):
            self.robot.detach(entity)
            self.robot.sleep(0.3)
        try:
            self.robot.execute_gripper(self.gripper.open_theta, 1.0, lambda: False)
        except Exception:
            pass
        self.robot.sleep(1.0)

    def _execute_steps(self, plan, _unused) -> bool:
        held = None
        released, rel_xyz = None, None
        for st in plan.steps:
            if self._aborted():
                self._drop_held(held)
                return False
            self.say(f'  > {st.name}')
            if st.kind == 'arm':
                if not self.robot.execute_arm(st, self._aborted):
                    self.say(f'  arm step "{st.name}" failed: {getattr(self.robot, "last_error", "")}')
                    self._drop_held(held)
                    return False
                if st.name == 'retreat' and released and rel_xyz is not None:
                    # the object must stay where it was released; if it rose with the gripper it is still stuck
                    poses = self.robot.entity_poses()
                    if poses and released in poses and poses[released][2] > rel_xyz[2] + 0.04:
                        self.say(f'  {released} is still stuck to the gripper (z {poses[released][2]:.2f} m) - '
                                 'releasing it again')
                        self._release_again(released)
            elif st.kind == 'gripper':
                if not self.robot.execute_gripper(st.theta, st.duration, self._aborted):
                    self._drop_held(held)
                    return False
            elif st.kind == 'attach':
                self.robot.attach(st.entity)
                held = st.entity
                self.robot.sleep(st.duration)
            elif st.kind == 'detach':
                poses = self.robot.entity_poses()
                rel_xyz = poses[st.entity][:3].copy() if poses and st.entity in poses else None
                for _ in range(3):                      # a single message can be missed by a slow simulator
                    self.robot.detach(st.entity)
                    self.robot.sleep(0.2)
                released = st.entity
                held = None
                self.robot.sleep(st.duration)
            elif st.kind == 'wait':
                self.robot.sleep(st.duration)
        return True

    def _drop_held(self, entity) -> None:
        """A failed move must never carry the object around: let it go where it is and open the fingers."""
        if not entity:
            return
        self.say(f'  releasing {entity} at its current position')
        for _ in range(3):
            self.robot.detach(entity)
            self.robot.sleep(0.3)
        try:
            self.robot.execute_gripper(self.gripper.open_theta, 1.0, lambda: False)
        except Exception:
            pass

    def _verify(self, label: Optional[str], xy0, entity: str, target: str) -> bool:
        poses = self.robot.entity_poses()
        if poses and entity in poses and target == 'bin':
            p = poses[entity]
            b = self.places['bin']
            inside = (abs(p[0] - b['center'][0]) < b['half_size'][0] + 0.03 and
                      abs(p[1] - b['center'][1]) < b['half_size'][1] + 0.03 and p[2] < b['rim_z'] + 0.05)
            if not inside:
                self.say(f'  ground truth: {entity} is at {np.round(p, 2)} - not in the bin')
            return bool(inside)
        res = self.observe()
        d = self._find(res, label, xy0)
        if d is None:
            return True
        return math.hypot(d.center[0] - xy0[0], d.center[1] - xy0[1]) > 0.05
