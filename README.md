# DeepRL


Deep Reinforcement learning with attention and recurrency for POMDPs.

Implemented algorithms in project:
* (Double/Dueling) Deep Recurrent Q-Learning (DRQN)
* (Double/Dueling) Deep Recurrent Q-Learning with Spatial Attention 
* (Double/Dueling) Deep Recurrent Q-Learning with Temporal Attention 
* (Double/Dueling) Deep Recurrent Q-Learning with Spatio-Temporal Attention 

Refer to https://github.com/ShangtongZhang/DeepRL if any other algorithm other than the ones above are used. 

*Acknowledgments*: Majority of work done in private gitlab repo of Loughborough University Comp Sci Department where most compute processing took place. 

# Dependency
* MacOS 10.12 or Ubuntu 16.04
* PyTorch v0.4.0
* Python 3.6, 3.5
* OpenAI Baselines (commit 8e56dd)
* Core dependencies: `pip install -e .`

# Usage

```examples.py``` contains examples for all the implemented algorithms

```Dockerfile``` contains a perfect environment, highly recommended 



# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
* [Hybrid Reward Architecture for Reinforcement Learning](https://arxiv.org/abs/1706.04208)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
* [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
* Some hyper-parameters are from [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), [OpenAI Baselines](https://github.com/openai/baselines) and [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
