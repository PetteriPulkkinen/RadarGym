import gym
from gym import spaces
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D
import numpy as np


class RevisitIntervalDiscrete(gym.Env):
    def __init__(self, sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10, n_obs=10, g_low=0.1, g_high=1):
        self.sim = sim
        self.p_loss = p_loss
        self.ri_min = ri_min
        self.ri_max = ri_max
        self.n_act = n_act
        self.n_obs = n_obs
        self.g_low = g_low
        self.g_high = g_high

        self.revisit_interval = 1
        self.angle_error = 0

        self.observation_space = spaces.Discrete(self.n_obs)  # Discretized theta values
        self.action_space = spaces.Discrete(self.n_act)  # Discretized revisit interval values

        super().__init__()

    def reset(self):
        self.sim.reset()
        return self._observation(update_successful=True)

    def step(self, action):
        self.revisit_interval = self.action2ri(action)
        update_successful, _, trajectory_ends = self.sim.step(self.revisit_interval)
        obs = self._observation(update_successful)
        reward = self._reward(update_successful, self.revisit_interval)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def _observation(self, update_successful):
        if update_successful:
            pos = self.sim.target.position
            pos_est = self.sim.radar.H @ self.sim.tracker.x.flatten()

            self.angle_error = pos_to_angle_error_2D(pos, pos_est)
            return self._discretize(self.angle_error, is_lost=False)
        else:
            return self._discretize(np.pi, is_lost=True)

    def _reward(self, update_successful, revisit_interval):
        if update_successful:
            return - 1 / revisit_interval
        else:
            return - self.p_loss

    def action2ri(self, action):
        return int((action * self.ri_max + (self.n_act - action - 1) * self.ri_min) / (self.n_act - 1))

    def _discretize(self, theta, is_lost):
        b_low = self.g_low * self.sim.radar.beamwidth
        b_high = self.g_high * self.sim.radar.beamwidth

        if is_lost:
            return self.n_obs - 1

        n = np.arange(0, self.n_obs - 2)
        b = b_low
        a = (b_high - b_low) / (self.n_obs - 3)

        f = a * n + b

        booleans = (f >= theta)
        if booleans.sum() == self.n_obs - 2:
            return 0
        elif booleans.sum() == 0:
            return self.n_obs - 2
        else:
            return np.argmin(booleans == False)
