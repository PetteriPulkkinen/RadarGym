import gym
from gym import spaces
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D
from trackingsimpy.simulation.revisit_interval import BaselineKalman
import numpy as np


class BaseRevisitInterval(gym.Env):
    def __init__(self, sim, p_loss, ri_min, ri_max, n_act):
        super(BaseRevisitInterval, self).__init__()
        self.sim = sim
        self.p_loss = p_loss
        self.ri_min = ri_min
        self.ri_max = ri_max
        self.n_act = n_act

        self.angle_error = 0
        self.alpha = 0.994

        self.action_space = spaces.Discrete(self.n_act)  # Discretized revisit interval values

    def reset(self):
        self.sim.reset()
        self.angle_error = 0
        return self._observation(update_successful=True, revisit_interval=1)

    def step(self, action):
        revisit_interval = self.action2ri(action)
        update_successful, n_missed, trajectory_ends = self.sim.step(revisit_interval)
        obs = self._observation(update_successful, revisit_interval)
        reward = self._reward(update_successful, revisit_interval, n_missed)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def action2ri(self, action):
        return int((action * self.ri_max + (self.n_act - action - 1) * self.ri_min) / (self.n_act - 1))

    def _observation(self, update_successful, revisit_interval):
        raise NotImplementedError

    def _reward(self, update_successful, revisit_interval, n_missed):
        if update_successful:
            return - (1 + n_missed) / revisit_interval
        else:
            return - self.p_loss


class RevisitIntervalDiscrete(BaseRevisitInterval):

    def __init__(self,
                 sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10, n_obs=10, g_low=0.1, g_high=1, is_pomdp=False,
                 noisy_obs=False):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.n_obs = n_obs
        self.g_low = g_low
        self.g_high = g_high
        self.is_pomdp = is_pomdp
        self.noisy_obs = noisy_obs

        self.observation_space = spaces.Discrete(self.n_obs)  # Discretized theta values

    def _observation(self, update_successful, revisit_interval):
        if update_successful:
            pos = self.sim.target.position
            pos_est = self.sim.radar.H @ self.sim.tracker.x.flatten()

            if self.is_pomdp:
                angle_error = self.sim.computer.theta
                self.angle_error = \
                    self.angle_error * self.alpha**revisit_interval + (1-self.alpha**revisit_interval) * angle_error
                angle_obs = np.abs(self.angle_error)
            else:
                if self.noisy_obs:
                    angle_obs = np.abs(self.sim.computer.theta)
                else:
                    angle_obs = np.abs(pos_to_angle_error_2D(pos, pos_est))
            return self._discretize(angle_obs, is_lost=False)
        else:
            return self._discretize(np.pi, is_lost=True)

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


class MMRevisitIntervalDiscrete(BaseRevisitInterval):

    def __init__(self,
                 sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10, n_obs=10, g_low=0.1, g_high=1):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.n_obs = n_obs
        self.g_low = g_low
        self.g_high = g_high

        self.observation_space = spaces.Discrete(self.n_obs)  # Discretized mu values

    def _observation(self, update_successful, revisit_interval):
        if update_successful:
            mu = self.sim.tracker.mu[0]
            return self._discretize(mu, is_lost=False)
        else:
            return self._discretize(0, is_lost=True)

    def _discretize(self, mu, is_lost):
        if is_lost:
            return self.n_obs - 1

        n = np.arange(0, self.n_obs - 2)
        b = self.g_low
        a = (self.g_high - self.g_low) / (self.n_obs - 3)

        f = a * n + b

        booleans = (f >= mu)
        if booleans.sum() == self.n_obs - 2:
            return 0
        elif booleans.sum() == 0:
            return self.n_obs - 2
        else:
            return np.argmin(booleans == False)


class RevisitIntervalContinuous(BaseRevisitInterval):
    def __init__(self, sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.observation_space = spaces.Box(low=0, high=np.pi, shape=(1,), dtype=float)

    def _observation(self, update_successful, revisit_interval):
        if update_successful:
            pos = self.sim.target.position
            pos_est = self.sim.radar.H @ self.sim.tracker.x.flatten()

            self.angle_error = np.abs(pos_to_angle_error_2D(pos, pos_est))
            return np.array([self.angle_error])
        else:
            return np.array([np.pi])


class RevisitIntervalBenchmarkDiscrete(RevisitIntervalDiscrete):
    def __init__(self, sims, p_loss, ri_min, ri_max, n_act, n_obs, g_low, g_high):
        super().__init__(
            sim=None,
            p_loss=p_loss,
            ri_min=ri_min,
            ri_max=ri_max,
            n_act=n_act,
            n_obs=n_obs,
            g_low=g_low,
            g_high=g_high
        )
        self._sims = sims
        self._traj_idx = None
        self._freeze = False

    def reset(self):
        if not self._freeze:
            self._traj_idx = np.random.randint(6)
        self.sim = self._sims[self._traj_idx]

        return super().reset()

    def freeze(self, traj_idx):
        self._traj_idx = traj_idx
        self.sim = self._sims[self._traj_idx]
        self._freeze = True

    def unfreeze(self):
        self._freeze = False


class MMRevisitIntervalBenchmarkDiscrete(MMRevisitIntervalDiscrete):
    def __init__(self, sims, p_loss, ri_min, ri_max, n_act, n_obs, g_low, g_high):
        super().__init__(
            sim=None,
            p_loss=p_loss,
            ri_min=ri_min,
            ri_max=ri_max,
            n_act=n_act,
            n_obs=n_obs,
            g_low=g_low,
            g_high=g_high
        )
        self._sims = sims
        self._traj_idx = None
        self._freeze = False

    def reset(self):
        if not self._freeze:
            self._traj_idx = np.random.randint(6)
        self.sim = self._sims[self._traj_idx]

        return super().reset()

    def freeze(self, traj_idx):
        self._traj_idx = traj_idx
        self.sim = self._sims[self._traj_idx]
        self._freeze = True

    def unfreeze(self):
        self._freeze = False
