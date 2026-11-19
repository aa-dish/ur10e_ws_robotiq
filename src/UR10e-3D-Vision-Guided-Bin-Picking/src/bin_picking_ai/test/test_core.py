"""Offline tests (no ROS / Gazebo needed):  cd src/bin_picking_ai && python3 -m pytest -q"""
import math
import os

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GZ = os.path.join(PKG, '..', 'gazebo')

from bin_picking_ai.geometry import (matrix_to_quat, quat_to_matrix, rpy_to_matrix, matrix_to_rpy,
                                     top_down_orientation, make_T)
from bin_picking_ai.kinematics import UR10e
from bin_picking_ai.llm import CommandParser, resolve_targets
from bin_picking_ai.vision import Catalog, Detector


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = rpy_to_matrix(*rng.uniform(-3, 3, 3))
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)
        assert np.allclose(rpy_to_matrix(*matrix_to_rpy(R)), R, atol=1e-9)


def test_top_down_orientation_is_rotation():
    R = top_down_orientation(0.7)
    assert np.allclose(R.T @ R, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9
    assert np.allclose(R[:, 2], [0, 0, -1])


def test_ik_roundtrip():
    r = UR10e(0.15)
    q = np.array([0.4, -1.2, 1.5, -1.9, -1.5, 0.3])
    q2 = r.solve(r.fk(q), q)
    assert q2 is not None and np.allclose(r.fk(q2), r.fk(q), atol=1e-3)


def test_home_pose_points_up():
    r = UR10e(0.15)
    T = r.fk([0, -1.57, 0, -1.57, 0, 0])
    assert T[2, 3] > 1.3


@pytest.fixture(scope='module')
def scene():
    from bin_picking_ai.tools.synthetic_scene import SyntheticScene
    return SyntheticScene(GZ, points_per_m2=4e5)


def test_vision_finds_and_names_every_object(scene):
    from bin_picking_ai.tools.synthetic_scene import ROBOT_BASE_WORLD
    cat = Catalog(os.path.join(PKG, 'data', 'catalog.json'))
    rgb, depth = scene.render()
    res = Detector(cat).detect(rgb, depth, scene.K, scene.T_base_cam)
    assert abs(res.table_z - 0.127) < 0.004
    assert len(res.detections) == len(scene.objects)
    ent2cls = {e: c for c, en in cat.entries.items() for e in en.entities}
    for d in res.detections:
        o = min(scene.objects, key=lambda o: np.hypot(*(np.array(o['pose'][:2]) - ROBOT_BASE_WORLD[:2] - d.center[:2])))
        assert d.label == ent2cls[o['name']], (d.label, o['name'])


def test_command_parsing_rules():
    cat = Catalog(os.path.join(PKG, 'data', 'catalog.json'))
    p = CommandParser(cat, None, use_llm=False)
    c = p.parse('pick the mustard bottle and put it in the bin', [])
    assert c.action == 'pick_place' and c.target == 'bin' and 'mustard' in c.object_phrase
    assert p.parse('go home', []).action == 'home'
    assert p.parse('stop', []).action == 'stop'
    assert p.parse('move the cereal to the left side of the table', []).target == 'table_left'
    assert p.parse('hello', []).action == 'unknown'


def test_end_to_end_pick_graspable(scene, monkeypatch):
    monkeypatch.delenv('UR_GRIPPER', raising=False)
    from bin_picking_ai.tools.offline_demo import make_system
    sysm, robot = make_system(gazebo_dir=GZ, points_per_m2=4e5)
    sysm.handle('pick the coffee mix and put it in the bin')
    p = robot.entity_poses()['nescafe_momento']
    assert np.all(np.abs(p[:2] - robot.bin_center) < robot.bin_half)


def test_2f140_picks_mustard(scene, monkeypatch):
    """Default gripper (Robotiq 2F-140, 140 mm stroke) can grasp the 116 mm mustard bottle."""
    monkeypatch.delenv('UR_GRIPPER', raising=False)
    from bin_picking_ai.tools.offline_demo import make_system
    sysm, robot = make_system(gazebo_dir=GZ, points_per_m2=4e5)
    sysm.handle('pick the mustard bottle and put it in the bin')
    p = robot.entity_poses()['mustard_bottle']
    assert np.all(np.abs(p[:2] - robot.bin_center) < robot.bin_half)


def test_2f140_clear_table(scene, monkeypatch):
    monkeypatch.delenv('UR_GRIPPER', raising=False)
    from bin_picking_ai.tools.offline_demo import make_system
    sysm, robot = make_system(gazebo_dir=GZ, points_per_m2=4e5)
    msg = sysm.handle('put all objects one by one in the bin')
    assert 'Done: 14/14' in msg


def test_oversize_object_refused(scene, monkeypatch):
    """With the Robotiq 2F-85 (UR_GRIPPER=2f85) the 116 mm mustard bottle exceeds the stroke: refused, left on the table."""
    monkeypatch.setenv('UR_GRIPPER', '2f85')
    from bin_picking_ai.tools.offline_demo import make_system
    sysm, robot = make_system(gazebo_dir=GZ, points_per_m2=4e5)
    sysm.handle('pick the mustard bottle and put it in the bin')
    p = robot.entity_poses()['mustard_bottle']
    assert not np.all(np.abs(p[:2] - robot.bin_center) < robot.bin_half)
    assert p[2] > 0.05
