from collections import deque

import numpy as np
import gym
from gym import spaces
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
import tensorflow as tf

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self._elapsed_steps = env._elapsed_steps

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self._elapsed_steps = env._elapsed_steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: ([int] or [float]) the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env, skip=4):
#         """
#         Return only every `skip`-th frame (frameskipping)
#
#         :param env: (Gym Environment) the environment
#         :param skip: (int) number of `skip`-th frame
#         """
#         gym.Wrapper.__init__(self, env)
#         # most recent raw observations (for max pooling across time steps)
#         self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
#         self._skip = skip

class MaxAndSkipEnv(gym.Env):
    def __init__(self, env, skip=4):
        """
      #         Return only every `skip`-th frame (frameskipping)
      #
      #         :param env: (Gym Environment) the environment
      #         :param skip: (int) number of `skip`-th frame
      #         """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip
        self.ale = self.env.ale
        self.env.reset() #needed for _elapsed_steps to be initialized
        self._elapsed_steps = self.env._elapsed_steps

    def clone_full_state(self):
        return self.env.clone_full_state()

    def restore_full_state(self, state):
        return self.env.restore_full_state(state)

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        return self.env.restore_state(state)

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.

        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)
        self._elapsed_steps = env._elapsed_steps

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward: (float)
        """
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)
        self._elapsed_steps = env._elapsed_steps

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)
        
        self._elapsed_steps = env._elapsed_steps

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames, f"{len(self.frames)} != {self.n_frames}"
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)
        self._elapsed_steps = env._elapsed_steps
        
    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class DelayWrapper(gym.Env):
    def __init__(self, env, delay_value, clone_full_state):
        self.orig_env = env
        self.delay_value = delay_value
        self.pending_actions = deque()
        self.stored_init_state = None
        self.is_atari_env = True
        self.action_space = self.orig_env.action_space
        self.observation_space = self.orig_env.observation_space
        self.clone_full_state = clone_full_state

    def step(self, action):
        if self.delay_value > 0:
            self.pending_actions.append(action)
            if len(self.pending_actions) - 1 >= self.delay_value:
                executed_action = self.pending_actions.popleft()
            else:
                executed_action = np.random.choice(self.action_space.n)
        else:
            executed_action = action
        return self.orig_env.step(executed_action)

    def reset(self):
        self.pending_actions.clear()
        return self.orig_env.reset()

    def get_pending_actions(self, pretrained_model, sess):
        if len(self.pending_actions) == 0 and self.delay_value > 0:
            # reconstruct anticipated trajectory using the oracle
            self.store_initial_state()
            curr_state = self.get_curr_state()
            for i in range(self.delay_value):
                # estimated_action = self._pretained_act(pretrained_model, sess, curr_state)
                estimated_action = np.random.choice(self.action_space.n)
                self.pending_actions.append(estimated_action)
                # curr_state = self.get_next_state(state=None, action=estimated_action)
                if curr_state is None:
                    break
            self.restore_initial_state()

        return self.pending_actions

    def _pretained_act(self, pretrained_model, sess, obs):
        feed_dict = {pretrained_model.obs_ph: np.expand_dims(obs, axis=0)}
        best_act_op = tf.argmax(pretrained_model.q_values, axis=1)
        best_act = sess.run(best_act_op, feed_dict)[0]
        return best_act

    def store_initial_state(self):
        if self.is_atari_env:
            if self.clone_full_state:
                self.stored_init_state = self.orig_env.clone_full_state()
            else:
                self.stored_init_state = self.orig_env.clone_state()
                # print("store!")
        else:
            self.stored_init_state = self.orig_env.unwrapped.state
    
    def store_initial_state_full(self):
        if self.is_atari_env:
            self.stored_init_state_full = self.orig_env.clone_full_state()
        else:
            self.stored_init_state = self.orig_env.unwrapped.state

    def restore_initial_state(self):
        if self.is_atari_env:
            if self.clone_full_state:
                self.orig_env.restore_full_state(self.stored_init_state)
            else:
                self.orig_env.restore_state(self.stored_init_state)
                # print("restore")
        else:
            self.orig_env.unwrapped.state = self.stored_init_state
    
    def restore_initial_state_full(self):
        if self.is_atari_env:
            self.orig_env.restore_full_state(self.stored_init_state_full)
        else:
            self.orig_env.unwrapped.state = self.stored_init_state

    def get_curr_state(self):
        if self.is_atari_env:
            curr_state = self.orig_env.ale.getScreenRGB2()
        else:
            curr_state = self.orig_env.unwrapped.state
        return curr_state

    def get_next_state(self, state, action):
        next_state, _, done, _ = self.orig_env.step(action)
        self.orig_env._elapsed_steps -= 1
        if done:
            return None
        return next_state

class StickyActionEnv(gym.Wrapper):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]
        
        self._elapsed_steps = env._elapsed_steps

    def reset(self, **kwargs):
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)


    def step(self, action: int):
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        tmp =  self.env.step(self._sticky_action)
        tmp[-1]['sticky_action'] = self._sticky_action
        return tmp

class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to np.ndarray before being passed to the model.

        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_atari(env_id, noop_reset=True, max_skip=True):
    """
    Create a wrapped atari Environment

    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    if max_skip:
        assert 'NoFrameskip' in env.spec.id
        env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False, sticky_action = False):
    """
    Configure environment for DeepMind-style Atari.

    :param env: (Gym Environment) the atari environment
    :param episode_life: (bool) wrap the episode life wrapper
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param frame_stack: (bool) wrap the frame stacking wrapper
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
        
    if sticky_action:
        env = StickyActionEnv(env, action_repeat_probability = sticky_action)
    return env



