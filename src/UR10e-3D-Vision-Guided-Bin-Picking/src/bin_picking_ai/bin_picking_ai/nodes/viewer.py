#!/usr/bin/env python3
"""Live camera window:  ros2 run bin_picking_ai viewer [topic]

Default topic is the annotated detection image.  Use /d435/image for the raw RGB camera.
"""
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class Viewer(Node):
    def __init__(self, topic):
        super().__init__('bin_picking_viewer')
        self.frame = None
        self.lock = threading.Lock()
        self.create_subscription(Image, topic, self.cb, 1)

    def cb(self, m: Image):
        ch = 3 if m.encoding in ('rgb8', 'bgr8') else 1
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch]
        a = a.reshape(m.height, m.width, ch) if ch == 3 else a
        if m.encoding == 'rgb8':
            a = a[:, :, ::-1]
        with self.lock:
            self.frame = np.ascontiguousarray(a)


def main(args=None):
    import cv2
    topic = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '/bin_picking/debug_image'
    rclpy.init(args=args)
    node = Viewer(topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f'Showing {topic}  (press q in the window to close)')
    while rclpy.ok():
        with node.lock:
            f = node.frame
        if f is not None:
            cv2.imshow('bin picking camera', f)
        if (cv2.waitKey(50) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
