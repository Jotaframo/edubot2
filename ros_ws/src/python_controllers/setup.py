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
            'example_pos_traj = python_controllers.example_pos_traj:main',
            'example_vel_traj = python_controllers.example_vel_traj:main',
            'pick_and_place = python_controllers.pick_place_open_loop:main',
            'constant_velocity = python_controllers.constant_velocity_follower:main',
            'constant_velocity_upward = python_controllers.constant_velocity_upward:main',
            'constant_velocity_follower = python_controllers.constant_velocity_follower:main',
            'pick_place_open_loop = python_controllers.pick_place_open_loop:main',
            'pick_place_roundtrip = python_controllers.pick_place_roundtrip:main',
            'pick_place_hw = python_controllers.pick_place_hw:main',
            'position_trajectory_follower = python_controllers.Position_Trajectory_Final:main',
            'pick_place_ez = python_controllers.pick_and_place_ez:main',
            'block_stacking_open_loop = python_controllers.block_stacking_open_loop:main',
            'block_stacking_hw = python_controllers.block_stacking_hw:main',
            'joint_pose_commander = python_controllers.joint_pose_commander:main',
        ],
    },
)
