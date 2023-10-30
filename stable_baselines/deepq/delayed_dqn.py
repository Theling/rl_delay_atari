from functools import partial

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
# from stable_baselines.deepq.build_graph import build_train; METHOD = "SMBS"
# from stable_baselines.deepq.build_graph_original import build_train; METHOD = "Delayed_Q"
from stable_baselines.deepq.policies import DQNPolicy
import wandb
from tqdm import tqdm
from collections import deque
from copy import deepcopy
from collections import OrderedDict

class DelayedDQN(OffPolicyRLModel):
    """
    The DQN model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, build_train, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False,
                 n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None,
                 delay_value=0, forward_model=None, load_pretrained_agent=True, is_delayed_agent=True,
                 is_delayed_augmented_agent=False, num_traj = 20):

        policy = partial(policy, is_delayed_agent=is_delayed_agent,
                         is_delayed_augmented_agent=is_delayed_augmented_agent, delay_value=delay_value)

        # TODO: replay_buffer refactoring
        super(DelayedDQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=DQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.build_train = build_train
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.delay_value = delay_value
        self.sample_buffer = deque()
        self.forward_model = forward_model
        
        self.num_traj = num_traj
        
        if self.forward_model is None:
            self.use_learned_forward_model = True
            #TODO: define and build forward_model here
            # keras_forward_model = self._build_model(loss='mse', input_size=self.state_size + 1, output_size=self.state_size)
            # self.forward_model = ForwardModel(keras_forward_model)
        else:
            self.use_learned_forward_model = False
            self.forward_model = forward_model
        self.load_pretrained_agent = load_pretrained_agent
        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None

        self.is_delayed_agent = is_delayed_agent
        self.is_delayed_augmented_agent = is_delayed_augmented_agent

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model, self.pretrained_model = self.build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                    double_q=self.double_q
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                # load pretrained agent that selects initial actions before the delay kicks in
                if self.load_pretrained_agent:
                    self.load_pretrained_model('pretrained_delay_0_step_1800000_reduced')

                self.summary = tf.summary.merge_all()

    def replay_buffer_delayed_add(self, obs_t, action, reward, obs_tp1, done, info):
        # Construct modified tuple by keeping old s_t with new a_{t+m}, r_{t+m} s_{t+m+1}
        new_tuple = [obs_t, action, reward, obs_tp1, done, info]
        self.sample_buffer.append(deepcopy(tuple(new_tuple)))
        if len(self.sample_buffer) - 1 >= self.delay_value:
            old_tuple = self.sample_buffer.popleft()
            # build time-coherent tuple from new tuple and old action
            new_tuple[1] = old_tuple[1]
            self.replay_buffer_add(*tuple(new_tuple))

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=self.exploration_initial_eps,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            # safe_mean_50 = lambda rewards: np.mean(episode_rewards) if len(episode_rewards) < 50 else np.mean(episode_rewards[-50:-1])
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for timestep in tqdm(range(total_timesteps)):
                # if timestep > int(1e6) and timestep % 200000 == 0:
                #     self.save_pretrained_model('pretrained_delay_0_step_' + str(timestep))
                    # self.load_pretrained_model('pretrained_delay_0_step_' + str(timestep))

                # Take action and update exploration to the newest value
                kwargs = {"num_traj": self.num_traj}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                if self.is_delayed_agent:
                    kwargs['use_learned_forward_model'] = self.use_learned_forward_model
                    kwargs['forward_model'] = self.forward_model
                with self.sess.as_default():
                    pending_actions = self.env.get_pending_actions(self.pretrained_model, self.sess)
                    if self.is_delayed_augmented_agent:
                        pending_actions_reshaped = np.expand_dims(pending_actions, axis=0)
                    else:
                        pending_actions_reshaped = []
                    action = self.act(np.array(obs)[None], update_eps=update_eps,
                                      pending_actions=pending_actions_reshaped,
                                      **kwargs)[0]
                    # print("obs: ", np.array(obs)[None].shape)
                    # print("q_values: ", q_values.shape)
                    
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                if self.is_delayed_augmented_agent:
                    obs_ = (obs_, pending_actions)
                    new_pending_actions = self.env.get_pending_actions(self.pretrained_model, self.sess)
                    new_obs_ = (new_obs_, new_pending_actions)
                # Store transition in the replay buffer.
                if self.is_delayed_agent:
                    self.replay_buffer_delayed_add(obs_, action, reward_, new_obs_, done, info)
                else:
                    self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)

                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                        self.num_timesteps)

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    self.sample_buffer.clear()
                    wandb.log({'episodic_reward': episode_rewards[-1]}, step=self.num_timesteps)
                    wandb.log({'mean_episodic_reward': np.mean(episode_rewards[-50:])}, step=self.num_timesteps)
                    # print('episodic_reward: {}'.format(episode_rewards[-1]))
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                               "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps),
                                                               env=self._vec_normalize_env)
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                                env=self._vec_normalize_env)
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        try:
                            if self.is_delayed_augmented_agent:
                                pending_t = np.asarray([obses_t[i][1] for i in range(obses_t.shape[0])])
                                obses_t = np.asarray([obses_t[i][0] for i in range(obses_t.shape[0])])
                                pending_tp1 = np.asarray([obses_tp1[i][1] for i in range(obses_tp1.shape[0])])
                                obses_tp1 = np.asarray([obses_tp1[i][0] for i in range(obses_tp1.shape[0])])
                            else:
                                pending_t = np.zeros((obses_t.shape[0], 0))
                                pending_tp1 = np.zeros((obses_t.shape[0], 0))
                            _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                            pending_t, pending_tp1, pending_tp1, sess=self.sess)
                        except Exception as e:
                            print('problem with pending_t: {}'.format(e))

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    max_mean_100ep_reward = mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 3)
                    if mean_100ep_reward > max_mean_100ep_reward:
                        max_mean_100ep_reward = mean_100ep_reward
                        callback.call('save')
                # print(update_eps)
                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("max mean 100 episode reward", max_mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

        callback.on_training_end()
        return self, episode_rewards

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def save_pretrained_model(self, save_path, cloudpickle=False):
        with self.graph.as_default():
            model_params = tf_util.get_globals_vars('deepq/model')
            parameter_values = self.sess.run(model_params)
            return_dictionary = OrderedDict((param.name, value) for param, value in zip(model_params, parameter_values))
            self._save_to_file(save_path, data=None, params=return_dictionary, cloudpickle=cloudpickle)

    def load_pretrained_model(self, save_path, cloudpickle=False):
        with self.graph.as_default():
            curr_pretrained_params = tf_util.get_globals_vars('deepq/pretrained_model')
            _, loaded_model_params = self._load_from_file(save_path)
            for model_param_name, model_param_val in loaded_model_params.items():
                for curr_pretrained_param in curr_pretrained_params:
                    if model_param_name.split('model')[1] in curr_pretrained_param.name:
                        self.sess.run(curr_pretrained_param.assign(model_param_val))
                        break
            # for (var_name, var_val), curr_pretrained_param in zip(loaded_pretrained_params.items(), curr_pretrained_params):
            #     assert curr_pretrained_param.name == var_name
            #     self.sess.run(curr_pretrained_param.assign(var_val))
            print('successfuly loaded pretrained network from: {}'.format(save_path))