**Status:** Active (under active development, breaking changes may occur)

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# Learn everything

This project is forked from openAI baseline, and experimenting on reinforcement learning algorithms.

We use this repo because the interface is relatively easy to use (not really, just realtively) and several reinforcement learning algorithm is already implemented.

## Code location

All changes in this project is located in `learn2optimize` folder and a few lines in `baselines/run.py`, other folders are inherited directly from openai baseline.

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```
Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```
will set entropy coefficient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each, and create a separate network for value function estimation (so that its parameters are not shared with the policy network, but the structure is the same)

See docstrings in [common/models.py](baselines/common/models.py) for description of network parameters for each type of model, and 
docstring for [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152) for the description of the ppo2 hyperparameters. 

### Example 2. DQN on Atari 
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## Saving, loading and visualizing models

### Saving and loading the model
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models. 
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively. 
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize: 
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*NOTE:* Mujoco environments require normalization to work properly, so we wrap them with VecNormalize wrapper. Currently, to ensure the models are saved with normalization (so that trained models can be restored and run without further training) the normalization coefficients are saved as tensorflow variables. This can decrease the performance somewhat, so if you require high-throughput steps with Mujoco and do not need saving/restoring the models, it may make sense to use numpy normalization instead. To do that, set 'use_tf=False` in [baselines/run.py](baselines/run.py#L116). 

### Logging and vizualizing learning curves and other training metrics
By default, all summary data, including progress, standard output, is saved to a unique directory in a temp folder, specified by a call to Python's [tempfile.gettempdir()](https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir).
The directory can be changed with the `--log_path` command-line option.
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2 --log_path=~/logs/Pong/
```
*NOTE:* Please be aware that the logger will overwrite files of the same name in an existing directory, thus it's recommended that folder names be given a unique timestamp to prevent overwritten logs.

Another way the temp directory can be changed is through the use of the `$OPENAI_LOGDIR` environment variable.

For examples on how to load and display the training data, see [here](docs/viz/viz.ipynb).

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2) 
- [TRPO](baselines/trpo_mpi)

In the root directory, run `python baselines/run.py` with arguments needed.

## Installation

- We would recommand installing with anaconda environment and tensorflow installed. For more details, please refer to [here](https://github.com/openai/baselines#installation)

- Please install `tensorflow_pribability` and upgrate `tensorflow` via pip after basic baselines requirement is met, since our project needs that.


Further readings, please refer to [OpenAI baselines](https://github.com/openai/baselines)
