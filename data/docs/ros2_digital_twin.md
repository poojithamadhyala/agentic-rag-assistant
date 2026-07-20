# ROS 2 Digital Twin Architecture

A digital twin mirrors a physical system in simulation, exchanging state over
ROS 2 topics, services, and actions using DDS middleware.

## Nodes and topics

Typical warehouse-twin nodes: a robot state publisher, a controller manager
running ros2_control, a MoveIt 2 motion-planning node for the UR5e arm, and a
gripper action server for the Robotiq 2F-85.

## QoS

Sensor streams use best-effort QoS with small history depth; command topics
use reliable QoS. Mismatched QoS profiles are the most common cause of
silent subscription failures in ROS 2.

## UR5e specifications

The UR5e has 6 revolute joints, a 5 kg payload, an 850 mm reach, and joint
speed up to 180 deg/s. Control interfaces include position, velocity, and
effort via ros2_control hardware interfaces at 500 Hz.
