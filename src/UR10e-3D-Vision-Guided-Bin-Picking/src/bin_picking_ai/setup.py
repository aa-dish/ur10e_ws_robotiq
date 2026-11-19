import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'bin_picking_ai'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'data'), glob('data/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='surabhi',
    maintainer_email='surabhi.gupta1617@gmail.com',
    description='Vision + LLM driven pick and place for the UR10e bin-picking simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bin_picking_node = bin_picking_ai.nodes.bin_picking_node:main',
            'chat = bin_picking_ai.nodes.chat:main',
            'viewer = bin_picking_ai.nodes.viewer:main',
            'gen_attach_xacro = bin_picking_ai.tools.gen_attach_xacro:main',
            'offline_demo = bin_picking_ai.tools.offline_demo:main',
        ],
    },
)
