#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''Variations with neuromodulation implemented at Loughborough University.'''

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
from deep_rl.agent.ApnnDQN_agent import ApnnDQNAgent
import os
from shutil import copy
from DynamicMazeEnv.gym_CTMaze.envs.CTMaze_plot import CTMaze_plot
from DynamicMazeEnv.gym_CTMaze.envs import CTMaze_env
from DynamicMazeEnv.gym_CTMaze.envs.CTMaze_conf import CTMaze_conf
from DynamicMazeEnv.gym_CTMaze.envs.CTMaze_images import CTMaze_images
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def dqn_maze(name):
    config = Config()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory="./DynamicMazeEnv/maze.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = "dqn_pa_gs_{0}_d_{1}_bf_{2}_cmaxl_{3}_cminl_{4}_".format(config.conf_data['general_seed'],\
                                                                             config.conf_data['maze_shape']['depth'],\
                                                                             config.conf_data['maze_shape']['branching_factor'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length']) +\
                                                                             name
    config.expID = "baseline"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(DQNAgent(config))

def qrdqn_maze(name):
    config = Config()
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory="./DynamicMazeEnv/maze.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = "dqn_pa_gs_{0}_d_{1}_bf_{2}_cmaxl_{3}_cminl_{4}_".format(config.conf_data['general_seed'],\
                                                                             config.conf_data['maze_shape']['depth'],\
                                                                             config.conf_data['maze_shape']['branching_factor'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length']) +\
                                                                             name
    config.expID = "qrdqn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            QuantileNet(action_dim, config.num_quantiles, NatureConvBody_lstm())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 1000
    config.exploration_steps= 15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(QuantileRegressionDQNAgent(config))

def qrdqn_maze_lstm(name):
    config = Config()
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory="./DynamicMazeEnv/maze.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = "dqn_pa_gs_{0}_d_{1}_bf_{2}_cmaxl_{3}_cminl_{4}_".format(config.conf_data['general_seed'],\
                                                                             config.conf_data['maze_shape']['depth'],\
                                                                             config.conf_data['maze_shape']['branching_factor'],\
                                                                             config.conf_data['maze_shape']['corridor_max_length'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length']) +\
                                                                             name
    config.expID = "qrdqn_lstm"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            QuantileNet(action_dim, config.num_quantiles, NatureConvBody_lstm())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(QuantileRegressionDQNAgent(config))

def att_drqn_maze(name):
    config = Config()
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory="./DynamicMazeEnv/maze.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = "dqn_pa_gs_{0}_d_{1}_bf_{2}_cmaxl_{3}_cminl_{4}_".format(config.conf_data['general_seed'],\
                                                                             config.conf_data['maze_shape']['depth'],\
                                                                             config.conf_data['maze_shape']['branching_factor'],\
                                                                             config.conf_data['maze_shape']['corridor_max_length'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length']) +\
                                                                             name
    config.expID = "qrdqn_lstm"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 4
    config.task_fn = lambda: Maze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            VanillaNet(action_dim,TempAttDRQNBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(DQNAgent(config))


def apnn_maze(name):
    config = Config()
    config.seed = 123456
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory="./DynamicMazeEnv/maze.json"
    apnn_config_file_directory="./apnn_parameters.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    apnn_configuration=CTMaze_conf(apnn_config_file_directory)
    config.conf_data = configuration.getParameters()
    config.apnn_conf_data=apnn_configuration.getParameters()
    print(config.conf_data)
    config.expType = "dqn_pa_gs_{0}_d_{1}_bf_{2}_cmaxl_{3}_cminl_{4}_".format(config.conf_data['general_seed'],\
                                                                             config.conf_data['maze_shape']['depth'],\
                                                                             config.conf_data['maze_shape']['branching_factor'],\
                                                                             config.conf_data['maze_shape']['corridor_max_length'],\
                                                                             config.conf_data['maze_shape']['corridor_min_length']) +\
                                                                             name
    config.expID = "apnn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            ApnnNet(action_dim, ApnnConvBody(),config.apnn_conf_data)
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json and apnn config file for future references
    copy(maze_conf_file_directory,config.log_dir)
    copy(apnn_config_file_directory,config.log_dir)
    #create debugging folder
    os.makedirs(config.log_dir+"/debug_matrix/")
    run_episodes(ApnnDQNAgent(config))


if __name__ == '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(0)
    #apnn_maze('CTMaze-v0')
    att_drqn_maze('CTMaze-v0')
    #dqn_pixel_atari('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_2l('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_diff('BreakoutNoFrameskip-v4')

#    mod_dqn_pixel_atari_3l_2sig('BreakoutNoFrameskip-v4')
#    mod_dqn_pixel_atari_3l_4sig('BreakoutNoFrameskip-v4')
#    mod_dqn_pixel_atari_3l_relu_shift1('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_relu6_shift05p05('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_diff_relu6_shift05p05('BreakoutNoFrameskip-v4')

#    mod_dqn_pixel_atari_3l_relu_shift05p05('BreakoutNoFrameskip-v4')

    #mod_dqn_pixel_atari_3l_diff('BreakoutNoFrameskip-v4')

    #quantile_regression_dqn_pixel_atari('BreakoutNoFrameskip-v4')
#    ddpg_pixel()
    #quantile_regression_dqn_pixel_atari_mod('BreakoutNoFrameskip-v4')
#    quantile_regression_dqn_pixel_atari_mod_surprise('BreakoutNoFrameskip-v4')

    #quantile_regression_dqn_pixel_atari_mod('RiverraidNoFrameskip-v0')
    #quantile_regression_dqn_pixel_atari_noframeskip('Riverraid-v4')

    #categorical_dqn_pixel_atari('BreakoutNoFrameskip-v4')
#    categorical_dqn_pixel_atari_mod('BreakoutNoFrameskip-v4')

    #ppo_pixel_atari('BreakoutNoFrameskip-v4')
    #ppo_pa_mod('BreakoutNoFrameskip-v4')

#    plot()
