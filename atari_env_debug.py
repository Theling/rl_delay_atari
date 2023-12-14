import gym
import wandb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnCnnPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import logger, A2C
from stable_baselines import DQN
from stable_baselines.deepq.policy_iteration import PI
from stable_baselines.deepq.delayed_dqn import DelayedDQN

from stable_baselines.common.atari_wrappers import make_atari, DelayWrapper, MaxAndSkipEnv, wrap_deepmind
from stable_baselines.common.callbacks import ModelSaveCallback
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, SubprocVecEnv
from functools import partial
import numpy as np
import sys
import os

# env_name = sys.argv[1] + '-v0'
env_name = "MsPacman-v0"
# env = gym.make('ALE/Breakout-v0')

METHOD = "SMBS" 
# METHOD = "Delayed_Q"
# METHOD = "AMDP"

if METHOD == "SMBS":
    from stable_baselines.deepq.build_graph import build_train
    agent_type = "delayed"
elif METHOD == "Delayed_Q":
    from stable_baselines.deepq.build_graph_original import build_train
    agent_type = "delayed"
elif METHOD == "AMDP":
    from stable_baselines.deepq.build_graph_original import build_train
    agent_type = "augmented"
else:
    raise


AVERAGE_OVER_LAST_EP = 0.05

def make_delayed_env(config):
    env = gym.make(config.env_name)
    env.reset()
    
    tmp = env.ale.getScreenRGB2().copy()
    clone_state = np.array(env.clone_state())
    
    obs, _, _, _ = env.step(1)
    obs2, _, _, _ = env.step(0)
    obs = env.ale.getScreenRGB2()
    print("should be different: ", abs(tmp - obs).sum())
    # print("should be different: ", abs(tmp - obs).sum())
    env.restore_state(clone_state)
    # print("should be 0", abs(clone_state - np.array(env.clone_state())).sum())
    obs = env.ale.getScreenRGB2()
    print("should be 0: ", abs(tmp - obs).sum())
    obs3, _, _, _ = env.step(1)
    # print("should be 0: ", abs(obs - obs3).sum())
    obs4, _, _, _ = env.step(0)
    # print("should be 0: ", abs(obs2 - obs4).sum())
    for _ in range(10):
        env.step(np.random.choice([0,1]))
        env.restore_state(clone_state)
    obs = env.ale.getScreenRGB2()
    print("should be 0: ", abs(tmp - obs).sum())
    # obs5, _, _, _ = env.step(1)
    # print("should be 0: ", abs(obs - obs5).sum()) 
    
    if config.deepmind_wrapper:
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True, sticky_action = config.sticky_action)
    else:
        env = MaxAndSkipEnv(env, skip=4)
        
    env = DelayWrapper(env, config.delay_value, config.clone_full_state)
    return env

AGENT_NAME = ''
import platform
if platform.system() == 'Darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

# TOTAL_TIMESTEPS = int(1e6)

hyperparameter_defaults = dict(
    train_freq=4,
    exploration_initial_eps=1.0,
    learning_rate=0.0001,
    target_network_update_freq=1000,
    exploration_final_eps=0.001,
    seed=1,
    env_name=env_name,#'Enduro-v0',#'MsPacman-v0', #'MsPacman-v0',
    gamma=0.99,
    delay_value=25,
    buffer_size=50000,
    prioritized_replay=True,
    # fixed_frame_skip=True,
    clone_full_state=False,
    load_pretrained_agent=False,
    agent_type=agent_type,#'delayed', #'delayed', 'augmented', 'oblivious', 'rnn'
    num_rnn_envs=4,
    deepmind_wrapper=True,
    total_timesteps=int(1e6),
    num_traj=20,
    sticky_action = 0.,
    exp_name = f"Method_{METHOD}",
)
# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="stable_baselines_tf-rl_delay-debug")
# wandb.log(hyperparameter_defaults, commit = False)
print(hyperparameter_defaults)
config = wandb.config


#TODO: check if using fixed 4-frame skip is better
# env = make_atari('BreakoutNoFrameskip-v4')
agent_full_name = wandb.run.id + '_' + \
                    AGENT_NAME + METHOD + '_' + \
                    env_name + \
                    f"_d{hyperparameter_defaults['delay_value']}"  + \
                    f"_r{hyperparameter_defaults['sticky_action']:.2f}" 
# Save a checkpoint every 1000 steps
checkpoint_callback = ModelSaveCallback(save_path=f'./logs_new/{agent_full_name}/',
                                         name_prefix='best')
# checkpoint_callback = None
# model = DQN(LnCnnPolicy, env, verbose=1, train_freq=config.train_freq, learning_rate=config.learning_rate,
#                 double_q=True, target_network_update_freq=config.target_network_update_freq,
#             gamma=config.gamma, prioritized_replay=True, exploration_initial_eps=config.exploration_initial_eps,
#             exploration_final_eps=config.exploration_final_eps)
if config.agent_type == 'rnn':
    # env = SubprocVecEnv([make_delayed_env(config) for i in range(config.num_rnn_envs)])
    env = DummyVecEnv([partial(make_delayed_env, config=config)])
    model = A2C(CnnLnLstmPolicy, env, verbose=1, gamma=config.gamma, tensorboard_log='')
else:
    env = make_delayed_env(config)
    if config.agent_type == 'delayed':
        is_delayed_agent = True
        is_delayed_augmented_agent = False
    elif config.agent_type == 'augmented':
        is_delayed_agent = False
        is_delayed_augmented_agent = True
    else: # 'oblivious'
        is_delayed_agent = False
        is_delayed_augmented_agent = False

#     model = DelayedDQN(LnCnnPolicy, env, build_train=build_train, verbose=1, train_freq=config.train_freq, learning_rate=config.learning_rate,
#                     double_q=True, target_network_update_freq=config.target_network_update_freq,
#                 gamma=config.gamma, prioritized_replay=config.prioritized_replay, exploration_initial_eps=config.exploration_initial_eps,
#                 exploration_final_eps=config.exploration_final_eps, delay_value=config.delay_value,
#                        forward_model=env, buffer_size=config.buffer_size, load_pretrained_agent=config.load_pretrained_agent,
#                        is_delayed_agent=is_delayed_agent, is_delayed_augmented_agent=is_delayed_augmented_agent, num_traj = config.num_traj)

# _, episode_rewards = model.learn(total_timesteps=config.total_timesteps, callback=checkpoint_callback)
# tot_ep_num = len(episode_rewards)
# avg_over = round(tot_ep_num * AVERAGE_OVER_LAST_EP)
# final_avg_score = np.mean(episode_rewards[-avg_over:])
# wandb.log({'final_score': final_avg_score})

# path = os.path.join(checkpoint_callback.save_path, 'final')
# model.save(path)

# del model # remove to demonstrate saving and loading
#
# model = DelayedDQN.load("deepq_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()