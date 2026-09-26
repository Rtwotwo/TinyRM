# TinyRM: Structured Reward and Recursive Self-Improvement Demo

This repository contains a compact reinforcement learning demo for a grid robot. The agent collects a key, avoids walls and hazards, and reaches a goal. The small environment makes reward learning, value estimation, and the recursive improvement loop easy to inspect. It is not a trained Unitree G1 controller.

## Architecture

- A structured reward ensemble learns six signed, context-conditioned components from synthetic trajectory preferences.
- Separate spatial actor and value networks are optimized with PPO and generalized advantage estimation.
- Each improvement round gathers new trajectories, asks the synthetic judge for uncertain preference labels, retrains the reward ensemble, trains a candidate policy, and accepts it only if held-out task evaluation improves.
- Expert demonstration states provide an adaptive reset curriculum. This is an auxiliary training technique; RSI here means Recursive Self-Improvement.

The judge supplies labels and acceptance scores. PPO uses the learned reward, not the judge's per-step score.

## Run in the yolo environment

From the repository root:

~~~powershell
conda run -n yolo python -m src.trainer.run_demo
~~~

To choose a device or a shorter run:

~~~powershell
conda run -n yolo python -m src.trainer.run_demo --device cuda --rounds 2 --rollout-steps 512
~~~

A run writes its configuration, checkpoints, metrics, independent evaluation, and learning plot under exp/rsi_demo/<run-id>/. The pre-existing run exp/rsi_demo/20260926_205844/ contains a completed example.

Evaluate a checkpoint on new seeds:

~~~powershell
conda run -n yolo python -m src.trainer.evaluate exp/rsi_demo/20260926_205844/champion_round_3.pt --episodes 64 --seed 140000
~~~

## Source layout

- src/config/demo.py: experiment parameters.
- src/envs/grid_robot.py: navigation environment and structured transition features.
- src/network/actor_critic.py: separate policy and value encoders.
- src/network/reward_model.py: interpretable reward ensemble.
- src/trainer/ppo.py: PPO rollout, GAE, and optimization.
- src/trainer/rsi.py: preference updates, candidate evaluation, and acceptance gate.
- src/trainer/reference_states.py: adaptive demonstration-state curriculum.

The G1 MuJoCo model and procedural dance reference generator remain available in agents/robots/unitree_g1/ and src/trainer/generate_g1_dance_motion.py. Generate a reference with:

~~~powershell
conda run -n yolo python -m src.trainer.generate_g1_dance_motion
~~~

Historical G1 experiment artifacts remain under exp/spring_dance/. The earlier G1 PPO training and evaluation entry points have been removed.
