# Unitree G1 Procedural Dance Tracking

A MuJoCo closed-loop PPO workflow for tracking a fixed, procedural G1 dance routine. The project borrows ideas from embodied-robotics examples, including Datawhale, but it does not extract dance motion from a Spring Festival Gala video and does not require PromptHMR, SMPL-X, GMR, or downloaded Datawhale model files.

## Control pipeline

The reference trajectory provides the beat-synchronized motion. PPO observes the G1 state and learns bounded torque corrections for the 29 actuated joints. A low-level PD controller and pelvis stabilizer support the simulation.

The 32-second choreography runs at 112 BPM and contains seven phrases: arm raise, step-touch and hip groove, diagonal punches, alternating knee lifts, overhead arm wave, fast footwork with double punches, and a wide final pose.

## Environment

Use the existing Conda environment named yolo. Install the project dependencies without replacing its PyTorch build:

~~~powershell
conda run -n yolo python -m pip install --no-deps -r requirements.txt
~~~

The G1 MuJoCo model is agents/robots/unitree_g1/g1_mocap_29dof.xml.

## Generate the dance reference

~~~powershell
conda run -n yolo python -m src.trainer.generate_g1_dance_motion
~~~

The generated reference and metadata are stored under exp/spring_dance/.

## Train PPO

~~~powershell
conda run -n yolo python -m src.trainer.train_g1_motion_tracking --total-timesteps 2000000 --n-envs 4 --device cuda
~~~

The policy optimizer uses CUDA when available. MuJoCo physics runs on the CPU. Checkpoints, final models, TensorBoard events, monitor CSV files, run configurations, and evaluation reports are kept under exp/spring_dance/.

## View training progress

~~~powershell
conda run -n yolo tensorboard --logdir exp/spring_dance/logs/tensorboard --port 6006
~~~

Open http://localhost:6006 and inspect total timesteps, FPS, mean episode reward, value loss, and explained variance.

## Evaluate and record a rollout

Pass a saved PPO checkpoint or final model to the evaluator:

~~~powershell
conda run -n yolo python -m src.trainer.evaluate_g1_motion_tracking --model exp/spring_dance/models/<model-file>.zip --episodes 1 --render-video
~~~

Evaluation summaries are saved under exp/spring_dance/evaluations/; rendered videos are saved under exp/spring_dance/videos/.

## Source layout

~~~text
agents/robots/unitree_g1/         G1 MJCF and mesh assets
src/config/                       Choreography and PPO settings
src/network/                      PPO actor/critic policy architecture
src/envs/                         Closed-loop G1 motion-tracking environment
src/trainer/                      Reference generation, training, and evaluation
exp/spring_dance/                 Models, checkpoints, logs, reports, and videos
~~~

The PPO implementation is provided by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The algorithm is described in Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).