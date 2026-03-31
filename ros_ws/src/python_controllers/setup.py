from setuptools import find_packages, setup

package_name = 'python_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anton',
    maintainer_email='a.bredenbeck@tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'workspace_visualizer = python_controllers.t01_workspace_visualizer:main',
            'position_trajectory_follower = python_controllers.t02_Position_Trajectory_Final:main',
            'joint_pose_commander = python_controllers.t02_joint_pose_commander:main',
            'constant_velocity_upward = python_controllers.t03_constant_velocity_upward:main',
            'pick_and_place = python_controllers.t04_pick_place_oneway:main',
            'pick_place_oneway = python_controllers.t04_pick_place_oneway:main',
            'pick_place_open_loop = python_controllers.t04_pick_place_oneway:main',
            'pick_place_roundtrip = python_controllers.t04_pick_place_roundtrip:main',
            'block_stacking = python_controllers.t05_block_stacking:main',
            'block_stacking_open_loop = python_controllers.t05_block_stacking:main',
            'example_pos_traj = python_controllers.example_pos_traj:main',
            'example_vel_traj = python_controllers.example_vel_traj:main',
        ],
    },
)
