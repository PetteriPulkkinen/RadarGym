import gym
from gym import spaces
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D
import numpy as np


def quantize(variable, nq, lb, ub):
    qz = int(np.floor((variable - lb)/(ub-lb) * (nq-2)))
    qz = qz + 1
    return np.max([0, np.min([qz, nq-1])])


def discretize_state(v, n):
    prd = np.cumprod(np.concatenate((np.array([1]), n[:-1])))
    return v @ prd


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
        return self._observation(update_successful=True, revisit_interval=0)  # Decodes track initiation

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
        obs1 = quantize(angle_obs, self.n_discretize, low_lim, high_lim)
        if self.multi_dim_obs:
            obs2 = self.action
            return discretize_state(np.array([obs1, obs2]), np.array([self.n_discretize, self.n_act]))
        else:

            return obs1


class MMRevisitIntervalDiscrete(BaseRevisitInterval):

    def __init__(self,
                 sim=None, p_loss=5000, ri_min=1, ri_max=100, n_act=10, n_discretize=10,
                 g_low=0.1, g_high=1, multi_dim_obs=False, range_obs=False):
        super().__init__(sim, p_loss, ri_min, ri_max, n_act)
        self.g_low = g_low
        self.g_high = g_high

        assert(multi_dim_obs == range_obs and range_obs == True)

        self.multi_dim_obs = multi_dim_obs
        self.range_obs = range_obs
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
        obs1 = quantize(mu, nq=self.n_discretize, lb=self.g_low, ub=self.g_high)
        if self.multi_dim_obs:
            obs2 = self.action
            return discretize_state(np.array([obs1, obs2]), np.array([self.n_discretize, self.n_act]))
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


class CommonRevisitInterval(gym.Env):
    def __init__(self, sims: object, c_loss: float, observations: dict, actions: dict, rewards: dict):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self._sims = sims

        self._freeze = False
        self._traj_idx = np.random.randint(len(self._sims))
        self.sim = self._sims[self._traj_idx]
        self.c_loss = c_loss

        self.n_obs = np.prod([value['nq'] for value in self.observations.values()]) + 2

        self.observation_space = spaces.Discrete(self.n_obs)
        self.action_space = spaces.Discrete(self.actions['n_acts'])

        self.action = None
        self.revisit_interval = 0

    def step(self, action):
        self.action = action
        k_ri = int(self._action2ri(action) / self.sim.DT)
        update_successful, n_missed, trajectory_ends = self.sim.step(k_ri)
        obs = self._observation(update_successful)
        reward = self._reward(update_successful, n_missed)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {'te': trajectory_ends, 'us': update_successful}

    def reset(self):
        if not self._freeze:
            self._traj_idx = np.random.randint(len(self._sims))
        self.sim = self._sims[self._traj_idx]

        self.sim.reset()
        self.action = 0
        self.revisit_interval = 0  # Decodes track initiation

        return self._observation(update_successful=True)

    def render(self, mode='human'):
        pass

    def _action2ri(self, action):
        act_type = self.actions['act_type']
        n = action
        tmax = self.actions['tmax']
        tmin = self.actions['tmin']
        n_acts = self.actions['n_acts']

        # Handle here different action spaces
        if act_type == 'direct':
            self.revisit_interval = (n * tmax + (n_acts - n - 1) * tmin) / (n_acts - 1)
        elif act_type == 'delta':
            dmax = self.actions['dmax']
            self.revisit_interval += (dmax * (2*n - n_acts + 1)) / (n_acts - 1)
            self.revisit_interval = np.max((tmin, np.min(tmax, self.revisit_interval)))
        else:
            raise NotImplementedError("The action type '{}' is not defined!".format(act_type))
        return self.revisit_interval

    def _observation(self, update_successful):
        if not update_successful and self.revisit_interval == 0:
            raise RuntimeError('Update not successful and revisit interval is 0.')
        if self.revisit_interval == 0:
            return self.n_obs - 2
        if not update_successful:
            return self.n_obs - 1

        types = list(self.observations.keys())
        n_variables = len(types)

        # Form the combined observation presentation
        self.qv = np.empty(shape=(n_variables,), dtype=int)
        self.nqv = np.empty(shape=(n_variables,), dtype=int)

        for idx, obs_type in enumerate(types):
            nq = self.observations[obs_type]['nq']
            lb = self.observations[obs_type]['lb']
            ub = self.observations[obs_type]['ub']
            self.nqv[idx] = nq

            # Handle here different observations
            if obs_type == 'mode_probability':
                nq = self.observations[obs_type]['nq']
                lb = self.observations[obs_type]['lb']
                ub = self.observations[obs_type]['ub']
                self.qv[idx] = quantize(self.sim.tracker.mu[0], nq, lb, ub)

            elif obs_type == 'innovation':
                pos = self.sim.target.position
                pos_est = self.sim.radar.H @ self.sim.tracker.x.flatten()
                angle_obs = np.abs(pos_to_angle_error_2D(pos, pos_est))
                self.qv[idx] = quantize(angle_obs, nq, lb, ub)

            elif obs_type == 'previous_ri':
                self.qv[idx] = quantize(self.revisit_interval, nq, lb, ub)

            elif obs_type == 'range':
                radius = np.linalg.norm(self.sim.radar.H @ self.sim.tracker.x.flatten())
                self.qv[idx] = quantize(radius, nq, lb, ub)
            else:
                raise NotImplementedError("The observation type '{}' is not defined!".format(obs_type))

        return discretize_state(self.qv, self.nqv)

    def _reward(self, update_successful, n_missed):
        if update_successful:
            return - (1 + n_missed) / self.revisit_interval * self.sim.radar.dwell_time
        else:
            return - self.c_loss

    def freeze(self, traj_idx):
        self._traj_idx = traj_idx
        self.sim = self._sims[self._traj_idx]
        self._freeze = True

    def unfreeze(self):
        self._freeze = False
