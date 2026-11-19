#!/usr/bin/env python3
"""Interactive terminal:  ros2 run bin_picking_ai chat      (type commands, Ctrl-D to quit)"""
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Chat(Node):
    def __init__(self):
        super().__init__('bin_picking_chat')
        self.pub = self.create_publisher(String, '/bin_picking/command', 10)
        self.create_subscription(String, '/bin_picking/status', lambda m: print('\r  robot> ' + m.data.replace('\n', '\n         '),
                                                                                  flush=True), 50)


def main(args=None):
    rclpy.init(args=args)
    node = Chat()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print('Type a command for the robot ("help" for examples, "stop" to interrupt, Ctrl-D to quit).')
    try:
        while rclpy.ok():
            try:
                text = input('you> ').strip()
            except EOFError:
                break
            if text:
                node.pub.publish(String(data=text))
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
