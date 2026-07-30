# Seb Naver ROS 2

## Overview

This repository provides a ROS 2 port of the SEB-Naver traversibility estimation system, designed for car-like robots operating on uneven terrain. It evaluates the terrain and estimates SE(2) costs using CUDA-accelerated processing, seamlessly integrating with Nav2 and Gazebo for robust local navigation. 

## Requirements

To run this project, you will need the following installed:
- **ROS 2 Humble**: The primary framework used for this port.
- **Gazebo**: Used for simulation and testing the navigation stack.
- **Nav2**: The ROS 2 Navigation framework.
- **CUDA Toolkit**: Required for the core terrain analyzer acceleration.

## Installation

1. Clone the repository into your ROS 2 workspace:
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/mathew4STAR/Seb-Naver-ros2.git
   ```

2. Install the necessary ROS 2 dependencies using `rosdep`:
   ```bash
   cd ~/ros2_ws
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

## Running the Project

To build the workspace and launch the simulation environment along with the navigation stack:

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch seb_naver_bringup bringup.launch.py
```

---

## About this Fork

This project adapts the traversibility estimation system from the original [seb_naver](https://github.com/zju-fast-lab/seb_naver) (based on the paper [*SEB-Naver*](https://arxiv.org/abs/2503.02412)). 

The original codebase was built in ROS 1 and utilized a custom physics/graphics simulator along with a complex custom low-level planner. This fork aims to modernize and standardize the approach by:
- Porting the entire framework to ROS 2 (Humble).
- Replacing the custom simulator with a Gazebo simulation environment.
- Rewriting the wrapper around the core CUDA terrain analyzer for ROS 2.
- Replacing the custom low-level planner with Nav2's MPPI. Since Nav2 cannot directly ingest SE(2) costs, a custom MPPI critic is used to query the SE(2) costs while simulating possible trajectories.
