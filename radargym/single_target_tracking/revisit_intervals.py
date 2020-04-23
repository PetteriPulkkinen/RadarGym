import gym
from gym import spaces
from trackingsimpy.simulation.revisit_interval import BaselineKalman
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D
import numpy as np
import matplotlib.pyplot as plt


class Renderer(object):
    """Renders target, trajectory, radar, measurements and predictions.
    """

    def __init__(self, simulation):
        self.figure = None
        self.ax = None  # scene
        self.ax_ri = None  # revisit interval
        self.ax_ae = None  # angular error
        self.target = None
        self.measurement = None
        self.prediction = None
        self.intervals = None
        self.theta_errors = None
        self.sim = simulation
        self.is_initialized = False

    def initialize(self):
        self.figure, axes = plt.subplots(3, 1)
        self.ax = axes[0]
        self.ax_ri = axes[1]
        self.ax_ae = axes[2]
        self.flush()
        plt.show(block=False)
        self.is_initialized = True

    def flush(self):
        self.ax.cla()
        self.ax_ri.cla()
        self.ax_ae.cla()

        self.ax.grid(True)
        self.ax_ri.grid(True)
        self.ax_ae.grid(True)

        self.ax.plot([0], [0], 'o', label='radar')
        self.ax.plot(self.sim.target.trajectory[:, 0], self.sim.target.trajectory[:, 2], '--', label='trajectory')

        self.intervals = self.ax_ri.plot(1, '-o')[0]

        self.theta_errors = self.ax_ae.plot(0, '-o')[0]

        pos = self._target_pos()
        pos_prior = self._prediction_pos()

        self.target = self.ax.plot(pos[0], pos[1], '*', label='target')[0]
        self.measurement = self.ax.plot(pos[0], pos[1], 'x', label='measurement')[0]  # No measurements at the beginning
        self.prediction = self.ax.plot(pos_prior[0], pos_prior[1], 'o', label='prediction')[0]

        self.ax.legend()

    def update(self, revisit_interval, angular_error):
        if not self.is_initialized:
            self.initialize()
        else:
            pos = self._target_pos()
            pos_prior = self._prediction_pos()
            z = self._measurement_pos()

            self.target.set_data(pos[0], pos[1])
            self.measurement.set_data(z[0], z[1])
            self.prediction.set_data(pos_prior[0], pos_prior[1])

            (x, y) = self.intervals.get_data()

            x = np.append(x, x.max()+1)
            y = np.append(y, revisit_interval)
            self.intervals.set_data(x, y)
            self.ax_ri.relim()
            self.ax_ri.autoscale_view()

            (x, y) = self.theta_errors.get_data()

            x = np.append(x, x.max() + 1)
            y = np.append(y, angular_error)
            self.theta_errors.set_data(x, y)
            self.ax_ae.relim()
            self.ax_ae.autoscale_view()

    def _target_pos(self):
        return self.sim.target.position

    def _prediction_pos(self):
        return self.sim.tracker.x_prior.flatten()[[0, 2]]

    def _measurement_pos(self):
        return self.sim.computer.z.flatten()


class RevisitIntervalDiscrete(gym.Env):
    RI_MAX = 100
    RI_MIN = 1
    N_ACTS = 10
    N_OBS = 10
    C_LOW = 0.1  # lower limit multiplier for beamwidth
    C_HIGH = 1  # higher limit multiplier for beamwidth
    P_LOSS = 5000

    def __init__(self):
        self.sim = BaselineKalman(
            traj_idx=4,
            var=1000,
            k_min=self.RI_MIN,
            k_max=self.RI_MAX
        )
        self.revisit_interval = 1
        self.angle_error = 0

        self.observation_space = spaces.Discrete(self.N_OBS)  # Discretized theta values
        self.action_space = spaces.Discrete(self.N_ACTS)  # Discretized revisit interval values

        self.renderer = Renderer(self.sim)

        super().__init__()

    def reset(self):
        if self.renderer.is_initialized:
            self.renderer.flush()
        self.sim.reset()
        return self._observation(update_successful=True)

    def step(self, action):
        self.revisit_interval = self.action2ri(action)
        update_successful, n_missed, trajectory_ends = self.sim.step(self.revisit_interval)
        obs = self._observation(update_successful)
        reward = self._reward(update_successful, n_missed, self.revisit_interval)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {}

    def render(self, mode='human'):
        self.renderer.update(self.revisit_interval, self.angle_error)
        self.renderer.figure.canvas.draw()

    def _observation(self, update_successful):
        if update_successful:
            pos = self.sim.target.position
            pos_est = self.sim.tracker.H @ self.sim.tracker.x.flatten()

            self.angle_error = pos_to_angle_error_2D(pos, pos_est)
            return self._discretize(self.angle_error, is_lost=False)
        else:
            return self._discretize(np.pi, is_lost=True)

    def _reward(self, update_successful, n_missed, revisit_interval):
        if update_successful:
            return - (n_missed + 1) / revisit_interval
        else:
            return - self.P_LOSS

    def action2ri(self, action):
        return int((action * self.RI_MAX + (self.N_ACTS - action - 1) * self.RI_MIN) / (self.N_ACTS - 1))

    def _discretize(self, theta, is_lost):
        b_low = self.C_LOW * self.sim.radar.beamwidth
        b_high = self.C_HIGH * self.sim.radar.beamwidth

        if is_lost:
            return self.N_OBS - 1

        n = np.arange(0, self.N_OBS - 2)
        b = b_low
        a = (b_high - b_low) / (self.N_OBS - 3)

        f = a * n + b

        booleans = (f >= theta)
        if booleans.sum() == self.N_OBS - 2:
            return 0
        elif booleans.sum() == 0:
            return self.N_OBS - 2
        else:
            return np.argmin(booleans == False)


class RevisitIntervalContinuous(gym.Env):
    RI_MAX = 100
    RI_MIN = 1
    N_ACTS = 10

    def __init__(self):
        self.sim = BaselineKalman(
            save=False,
            traj_idx=4,
            var=1000,
            k_min=self.RI_MIN,
            k_max=self.RI_MAX
        )

        self.revisit_interval = 1

        self.observation_space = spaces.Box(low=0, high=np.pi, shape=(1,), dtype=np.float)  # Angle and radius delta
        self.action_space = spaces.Discrete(self.N_ACTS)  # Discretized revisit interval values

        self.renderer = Renderer(self.sim)

        super().__init__()

    def reset(self):
        if self.renderer.is_initialized:
            self.renderer.flush()
        obs = self.sim.reset()
        return self._observation(update_successful=True)

    def step(self, action):
        self.revisit_interval = self._action2ri(action)
        update_successful, n_missed, trajectory_ends = self.sim.step(self.revisit_interval)
        obs = self._observation(update_successful)
        reward = self._reward(update_successful, n_missed, self.revisit_interval)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False
        return obs, reward, done, {}

    def render(self, mode='human'):
        self.renderer.update()
        self.renderer.figure.canvas.draw()

    def _observation(self, update_successful):
        if update_successful:
            H = self.sim.tracker_computer.tracker.H
            pos = H @ self.sim.target.x.flatten()
            pos_est = H @ self.sim.tracker_computer.tracker.x.flatten()

            angle_error = pos_to_angle_error_2D(pos, pos_est)
            return np.array([angle_error])
        else:
            return np.array([np.pi])

    def _reward(self, update_successful, n_missed, revisit_interval):
        if update_successful:
            return - (n_missed + 1) / revisit_interval
        else:
            return - self.sim.p_loss

    def _action2ri(self, action):
        return int((action * self.RI_MAX + (self.N_ACTS - action - 1) * self.RI_MIN) / (self.N_ACTS - 1))
