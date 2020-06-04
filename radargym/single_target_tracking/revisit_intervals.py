import gym
from gym import spaces
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D
import numpy as np


def discretize(value, N, low_lim, high_lim):
    n = np.arange(0, N-1)
    b = low_lim
    a = (high_lim - low_lim) / (N - 2)

    f = a * n + b

    booleans = (f > value)

    if booleans.sum() == N-1:
        return 0
    elif booleans.sum() == 0:
        return N-1
    else:
        return np.argmin((f > value) == False)


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
        self.action = 0

    def reset(self):
        self.sim.reset()
        self.angle_error = 0
        self.action = 0
        return self._observation(update_successful=True, revisit_interval=0)

    def step(self, action):
        self.action = action
        revisit_interval = self.action2ri(action)
        update_successful, n_missed, trajectory_ends = self.sim.step(revisit_interval)
        obs = self._observation(update_successful, revisit_interval)
        reward = self._reward(update_successful, revisit_interval, n_missed)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {'te': trajectory_ends, 'us': update_successful}

    def render(self, mode='human'):
        pass

    def action2ri(self, action):
        return int((action * self.ri_max + (self.n_act - action - 1) * self.ri_min) / (self.n_act - 1))

    def _observation(self, update_successful, revisit_interval):
        raise NotImplementedError

    def _reward(self, update_successful, revisit_interval, n_missed):
        if update_successful:
            return - (1 + n_missed) / revisit_interval * self.sim.radar.dwell_time / self.sim.DT
        else:
            return - self.p_loss


class RevisitIntervalDiscrete(BaseRevisitInterval):

    def __init__(self,
                 sim=None, p_loss=5000, ri_min=1, ri_max=100, n_discretize=10, n_act=10, g_low=0.1, g_high=1,
                 is_pomdp=False, noisy_obs=False, multi_dim_obs=False):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.g_low = g_low
        self.g_high = g_high
        self.is_pomdp = is_pomdp
        self.noisy_obs = noisy_obs
        self.multi_dim_obs = multi_dim_obs
        self.n_discretize = n_discretize

        if self.multi_dim_obs:
            self.n_obs = n_discretize * n_act + 2  # Discretized values * number of actions + initial + lost
        else:
            self.n_obs = self.n_discretize + 2  # Discretized values + initial + lost

        self.observation_space = spaces.Discrete(self.n_obs)  # Discretized theta values

    def _observation(self, update_successful, revisit_interval):
        if not update_successful and revisit_interval == 0:
            raise RuntimeError('Update not successful and revisit interval is 0.')
        if revisit_interval == 0:
            return self.n_obs - 2
        if not update_successful:
            return self.n_obs - 1

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

        low_lim = self.g_low * self.sim.radar.beamwidth
        high_lim = self.g_high * self.sim.radar.beamwidth
        obs1 = discretize(angle_obs, self.n_discretize, low_lim, high_lim)
        if self.multi_dim_obs:
            obs2 = self.action
            return obs1*self.n_discretize + obs2
        else:

            return obs1


class MMRevisitIntervalDiscrete(BaseRevisitInterval):

    def __init__(self,
                 sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10, n_discretize=10,
                 g_low=0.1, g_high=1, multi_dim_obs=False):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.g_low = g_low
        self.g_high = g_high

        self.multi_dim_obs = multi_dim_obs
        self.n_discretize = n_discretize

        if self.multi_dim_obs:
            self.n_obs = n_discretize * n_act + 2  # Discretized values * number of actions + initial + lost
        else:
            self.n_obs = self.n_discretize + 2  # Discretized values + initial + lost

        self.observation_space = spaces.Discrete(self.n_obs)  # Discretized mu values

    def _observation(self, update_successful, revisit_interval):
        if not update_successful and revisit_interval == 0:
            raise RuntimeError('Update not successful and revisit interval is 0.')
        if revisit_interval == 0:
            return self.n_obs - 2
        if not update_successful:
            return self.n_obs - 1

        mu = self.sim.tracker.mu[0]
        obs1 = discretize(mu, self.n_discretize, self.g_low, self.g_high)
        if self.multi_dim_obs:
            obs2 = self.action
            return obs1*self.n_discretize + obs2
        else:

            return obs1


class RevisitIntervalContinuous(BaseRevisitInterval):
    def __init__(self, sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.observation_space = spaces.Box(low=0, high=np.pi, shape=(1,), dtype=float)
        raise NotImplementedError

    def _observation(self, update_successful, revisit_interval):
        if update_successful:
            pos = self.sim.target.position
            pos_est = self.sim.radar.H @ self.sim.tracker.x.flatten()

            self.angle_error = np.abs(pos_to_angle_error_2D(pos, pos_est))
            return np.array([self.angle_error])
        else:
            return np.array([np.pi])


class RevisitIntervalBenchmarkDiscrete(RevisitIntervalDiscrete):
    def __init__(self, sims, p_loss, ri_min, ri_max, n_act, n_discretize, g_low, g_high, multi_dim_obs=False):
        super().__init__(
            sim=None,
            p_loss=p_loss,
            ri_min=ri_min,
            ri_max=ri_max,
            n_act=n_act,
            n_discretize=n_discretize,
            g_low=g_low,
            g_high=g_high,
            multi_dim_obs=multi_dim_obs
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
    def __init__(self, sims, p_loss, ri_min, ri_max, n_act, n_discretize, g_low, g_high, multi_dim_obs=False):
        super().__init__(
            sim=None,
            p_loss=p_loss,
            ri_min=ri_min,
            ri_max=ri_max,
            n_act=n_act,
            n_discretize=n_discretize,
            g_low=g_low,
            g_high=g_high,
            multi_dim_obs=multi_dim_obs
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
