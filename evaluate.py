import warnings
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")


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



import re

def extract_info_from_directory_name(directory_name):
    pattern = re.compile(r'(AMDP|SMBS|Delayed_Q)_(\w+)-v0_d(\d+)_r(\d+\.\d{2})_(\w+)')
    # pattern = re.compile(r'(AMDP)_(\w+)-v0_d(\d+)_r(\d+\.\d{2})_(\w+)')
    
    match = pattern.match(directory_name)

    if match:
        method_name, environment_name, delay_steps, randomness_factor, id = match.groups()
        delay_steps = int(delay_steps)
        randomness_factor = float(randomness_factor)
        environment_name += '-v0'
        return id, method_name, environment_name, delay_steps, randomness_factor
    else:
        raise ValueError(f"Invalid directory name format: {directory_name}.")
    
def is_evaluated(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name == "eval_results2.npy":
            return True
    return False

def load_newest_best_model(folder_path):
    best_model_pattern = re.compile(r'best_(\d+)_steps.zip')
    best_steps = 0
    best_model_path = None

    for file_name in os.listdir(folder_path):
        match = best_model_pattern.match(file_name)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_model_path = os.path.join(folder_path, file_name)

    if best_model_path:
        print(f'Loading best model with {best_steps} steps from {best_model_path}')
        return best_model_path
    else:
        print('No best model found in the specified folder.')
        return None
    
def load_final_model(folder_path):
    final_model_path = None
    for file_name in os.listdir(folder_path):
        if 'final' in file_name:
            final_model_path = os.path.join(folder_path, file_name)
            break

    if final_model_path:
        print(f'Loading final model from {final_model_path}')
        return final_model_path
    else:
        print('No best model found in the specified folder.')
        return None

# Example usage
load_path = "zgty2wc5_SMBS_MsPacman-v0_d25_r0.20"
log_dir = './logs_new2/'
load_best = False
load_path = sys.argv[1]
use_wandb = True

if is_evaluated(os.path.join(log_dir, load_path)): 
    quit()

id, METHOD, env_name, delay_steps, randomness_factor = extract_info_from_directory_name(load_path)
print(f'ID: {id}')
print(f'Method Name: {METHOD}')
print(f'Environment Name: {env_name}')
print(f'Delay Steps: {delay_steps}')
print(f'Randomness Factor: {randomness_factor}')


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
    delay_value=delay_steps,
    buffer_size=50000,
    prioritized_replay=True,
    # fixed_frame_skip=True,
    clone_full_state=False,
    load_pretrained_agent=False,
    agent_type=agent_type, #'delayed', 'augmented', 'oblivious', 'rnn'
    num_rnn_envs=4,
    deepmind_wrapper=True,
    total_timesteps=int(2e6),
    num_traj=50,
    sticky_action = randomness_factor,
    exp_name = f"Method_{METHOD}",
)
if use_wandb:
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, 
            project="stable_baselines_tf-rl_delay_test", 
            job_type='model_testing',
            tags=f"{METHOD}_d{delay_steps}_r{randomness_factor}",
            name=env_name+'_'+METHOD)
    # wandb.log(hyperparameter_defaults, commit = False)
else:
    wandb.init(config=hyperparameter_defaults)
print(hyperparameter_defaults)
config = wandb.config


#TODO: check if using fixed 4-frame skip is better
# env = make_atari('BreakoutNoFrameskip-v4')
# agent_full_name = wandb.run.id + '_' + \
#                     AGENT_NAME + METHOD + '_' + \
#                     env_name + \
#                     f"_d{hyperparameter_defaults['delay_value']}"  + \
#                     f"_r{hyperparameter_defaults['sticky_action']:.2f}" 
# # Save a checkpoint every 1000 steps
# checkpoint_callback = ModelSaveCallback(save_path=f'./logs/{agent_full_name}/',
#                                          name_prefix='best')
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

    if load_best:
        model_path = load_newest_best_model(os.path.join(log_dir, load_path))
        save_name = "eval_results.npy"
    else:
        model_path = load_final_model(os.path.join(log_dir, load_path))
        save_name = "eval_results2.npy"
    
    model = DelayedDQN.load(model_path, 
                            build_train, 
                            config, 
                            env = env,
                            is_delayed_agent = is_delayed_agent, 
                            is_delayed_augmented_agent = is_delayed_augmented_agent)
    # model = DelayedDQN(LnCnnPolicy, 
    #                    env, 
    #                    build_train=build_train, 
    #                    verbose=1, 
    #                    train_freq=config.train_freq, 
    #                    learning_rate=config.learning_rate,
    #                    double_q=True, 
    #                    target_network_update_freq=config.target_network_update_freq,
    #                     gamma=config.gamma, 
    #                     prioritized_replay=config.prioritized_replay, 
    #                     exploration_initial_eps=config.exploration_initial_eps,
    #                     exploration_final_eps=config.exploration_final_eps, 
    #                     delay_value=config.delay_value,
    #                    forward_model=env, 
    #                    buffer_size=config.buffer_size, 
    #                    load_pretrained_agent=config.load_pretrained_agent,
    #                    is_delayed_agent=is_delayed_agent, 
    #                    is_delayed_augmented_agent=is_delayed_augmented_agent, 
    #                    num_traj = config.num_traj)

# _, episode_rewards = model.learn(total_timesteps=config.total_timesteps, callback=checkpoint_callback)
# tot_ep_num = len(episode_rewards)
# avg_over = round(tot_ep_num * AVERAGE_OVER_LAST_EP)
# final_avg_score = np.mean(episode_rewards[-avg_over:])
# wandb.log({'final_score': final_avg_score})

# path = os.path.join(checkpoint_callback.save_path, 'final')
# model.save(path)

# del model # remove to demonstrate saving and loading
#

# quit()
menv = model.env
obs = menv.reset()
rew = [0.]
rew_step = []
for _ in tqdm(range(100000), desc='Evaluation', unit='iteration'):

    action, _states = model.predict(obs)
    obs, rewards, dones, info = menv.step(action)
    # print(rewards, dones)
    rew[-1] += rewards
    rew_step.append(rewards)
    if dones:
        wandb.log({'episode_reward': rew[-1]})
        # print(rew[-1])
        obs = menv.reset()
        rew.append(0.)

        # env.render()
        
wandb.log({'mean_reward': np.mean(rew), "reward_step": np.array(rew_step)})
np.save(os.path.join('./logs_new2/', load_path, save_name), rew)