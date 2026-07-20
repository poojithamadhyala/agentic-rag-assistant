# PPO Training for Legged Locomotion

Proximal Policy Optimization (PPO) is an on-policy reinforcement learning
algorithm widely used for locomotion tasks because of its stability and
simplicity.

## Key hyperparameters

Common starting points for humanoid locomotion: learning rate 3e-4, clip
range 0.2, discount factor gamma 0.99, GAE lambda 0.95, 2048-4096 steps per
rollout, and 10 epochs per update. Entropy bonus around 0.0 to 0.01
encourages exploration.

## Reward design

Locomotion rewards typically combine forward velocity tracking, alive bonus,
control-cost penalties, and joint-limit penalties. Explained variance close
to 1.0 indicates the value function predicts returns well; values above 0.9
suggest a healthy critic.

## Sim-to-real transfer

Domain randomization over mass, friction, motor strength, and sensor noise
improves transfer. Observation and action latency modeling is critical for
hardware deployment.
