"""Try commands without ROS / Gazebo:   python3 -m bin_picking_ai.tools.offline_demo "pick the mustard bottle" ..."""
import os
import sys

from ..orchestrator import BinPickingSystem, SystemConfig, gripper_config_file
from ..sim_robot import SimRobot
from ..llm import CommandParser, OllamaClient
from ..vision import Catalog
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_system(use_llm=False, gazebo_dir=None, points_per_m2=8e5):
    gazebo_dir = gazebo_dir or os.path.join(PKG, '..', 'gazebo')
    places = yaml.safe_load(open(os.path.join(PKG, 'config', 'places.yaml')))
    gfile = gripper_config_file(os.path.join(PKG, 'config'))
    tcp_z = yaml.safe_load(open(gfile))['gripper']['tcp_z']
    robot = SimRobot(gazebo_dir, places['home_q'], tcp_z=tcp_z, bin_center=places['bin']['center'],
                     bin_half=places['bin']['half_size'], points_per_m2=points_per_m2)
    cfg = SystemConfig(catalog_path=os.path.join(PKG, 'data', 'catalog.json'),
                       places_path=os.path.join(PKG, 'config', 'places.yaml'),
                       gripper_path=gfile,
                       world_objects_path=os.path.join(PKG, 'data', 'world_objects.json'))
    parser = CommandParser(Catalog(cfg.catalog_path), OllamaClient() if use_llm else None, use_llm=use_llm)
    return BinPickingSystem(robot, cfg, parser), robot


if __name__ == '__main__':
    sysm, robot = make_system(use_llm='--llm' in sys.argv)
    for c in [a for a in sys.argv[1:] if not a.startswith('--')] or ['what do you see']:
        print('\n>>>', c)
        sysm.handle(c)
