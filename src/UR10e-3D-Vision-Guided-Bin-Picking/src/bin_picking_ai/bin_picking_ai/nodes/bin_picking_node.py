#!/usr/bin/env python3
"""Main node: listens to /bin_picking/command (std_msgs/String), does the work, answers on /bin_picking/status.

ros2 topic pub --once /bin_picking/command std_msgs/msg/String "{data: 'pick the mustard bottle'}"
(or use the interactive terminal:  ros2 run bin_picking_ai chat)
"""
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ..llm import CommandParser, OllamaClient
from ..orchestrator import BinPickingSystem, SystemConfig, gripper_config_file
from ..vision import Catalog, VisionParams
from .ros_robot import RosRobot


class BinPickingNode(Node):
    def __init__(self):
        super().__init__('bin_picking')
        share = get_package_share_directory('bin_picking_ai')
        d = self.declare_parameter
        use_llm = d('use_llm', True).value
        host = d('ollama_host', 'http://localhost:11434').value
        model = d('ollama_model', 'auto').value
        model = '' if str(model).lower() in ('', 'auto', 'none') else str(model)
        left_right = d('left_right', 'image').value
        max_retries = d('max_retries', 2).value
        self.debug_rate = d('debug_image_rate', 2.0).value
        cfg = SystemConfig(
            catalog_path=os.path.join(share, 'data', 'catalog.json'),
            places_path=d('places_file', os.path.join(share, 'config', 'places.yaml')).value,
            gripper_path=d('gripper_file', gripper_config_file(os.path.join(share, 'config'))).value,
            world_objects_path=os.path.join(share, 'data', 'world_objects.json'),
            left_right=left_right, max_retries=int(max_retries))
        cat = Catalog(cfg.catalog_path)
        entities = sorted({e for c in cat.entries.values() for e in c.entities})
        self.robot = RosRobot(self, entities)
        ollama = OllamaClient(host, model) if use_llm else None
        parser = CommandParser(cat, ollama, use_llm=use_llm)
        self.get_logger().info('command understanding: ' + ('Ollama model ' + ollama.model if parser.llm_ready else 'rule-based'))
        self.system = BinPickingSystem(self.robot, cfg, parser)
        self.cmd_q: 'queue.Queue[str]' = queue.Queue()
        self.create_subscription(String, '/bin_picking/command', self._on_cmd, 10)
        self.dbg_pub = self.create_publisher(Image, '/bin_picking/debug_image', 1)
        self.scene_pub = self.create_publisher(String, '/bin_picking/scene', 1)
        self.create_timer(1.0 / max(self.debug_rate, 0.1), self._debug_tick)
        self._dbg_busy = False
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _on_cmd(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        if text.lower() in ('stop', 'halt', 'abort', 'cancel'):
            self.system.abort()
            self.robot.say('Stopping ...')
            return
        self.cmd_q.put(text)

    def _worker(self):
        if not self.robot.wait_ready(120):
            self.get_logger().error('controllers / joint states not available - are the arm_controller and '
                                    'gripper_controller spawned?')
        else:
            for _ in range(3):
                self.robot.release_all()
                time.sleep(0.3)
            time.sleep(3.0)
            if not self.robot.entity_poses():
                self.get_logger().warn('no object poses received on /world/empty/pose/info - releasing an object '
                                       'cannot be verified (check: ros2 topic hz /world/empty/pose/info)')
            self.robot.say('Ready. Moving to the home pose ...')
            try:
                self.system.go_home()
                self.robot.execute_gripper(self.system.gripper.open_theta, 1.0, lambda: False)
            except Exception as e:
                self.get_logger().error(f'homing failed: {e}')
            self.robot.say('Ready for commands. Try: "pick the mustard bottle and put it in the bin".')
        while rclpy.ok():
            try:
                text = self.cmd_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.get_logger().info(f'command: {text}')
            self.system.handle(text)

    # ---- debug image: what the robot sees, with names
    def _debug_tick(self):
        res = self.system.last_result
        if res is None or self._dbg_busy or self.dbg_pub.get_subscription_count() == 0:
            return
        self._dbg_busy = True
        try:
            img = cv2.cvtColor(res.color_map, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for dt in res.detections:
                x, y, w, h = [v * 2 for v in dt.bbox_px]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                name = self.system.catalog.display_name(dt.label) if dt.label else '?'
                cv2.putText(img, f'#{dt.id} {name}', (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(img, f'#{dt.id} {name}', (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            m = Image()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.height, m.width, m.encoding, m.step = img.shape[0], img.shape[1], 'bgr8', img.shape[1] * 3
            m.data = img.tobytes()
            self.dbg_pub.publish(m)
            s = String()
            s.data = json.dumps([dt.to_json() for dt in res.detections])
            self.scene_pub.publish(s)
        finally:
            self._dbg_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = BinPickingNode()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
