# MuJoCo Simulation Basics

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for
model-based control and robotics research. Models are described in MJCF, an
XML format that defines bodies, joints, actuators, sensors, and contact
properties.

## Timesteps and integration

The default simulation timestep in MuJoCo is 0.002 seconds (500 Hz). Smaller
timesteps improve stability for stiff contacts but increase compute cost. The
default integrator is semi-implicit Euler; RK4 is available for higher
accuracy at greater cost.

## Actuators

MuJoCo supports motor, position, velocity, and general actuators. Position
actuators apply a PD-style control law where kp sets the proportional gain.
Torque limits are set via ctrlrange or forcerange attributes on the actuator.

## Contacts

Contact dynamics use a soft-constraint model parameterized by solref and
solimp. Typical solref values are (0.02, 1), meaning a 20 ms time constant
and critical damping ratio of 1.
