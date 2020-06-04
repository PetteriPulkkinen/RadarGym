from gym.envs.registration import register
from trackingsimpy.simulation.revisit_interval import DefinedIMM, BaselineKalman, DefinedCVCAIMM
from trackingsimpy.simulation.revisit_interval import DefinedKalman, IMMBenchmark


register(
    id='Tracking-v0',
    entry_point='radargym.single_target_tracking:LinearTrackingDiscrete'
)

register(
    id='Tracking-v1',
    entry_point='radargym.single_target_tracking:LinearTrackingSmooth'
)

register(
    id='revisit-v00',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=(4.5 * 9.81) ** 2),
        'p_loss': 5000,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 1,
        'ri_max': 500,
    }
)

register(
    id='revisit-v01',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': DefinedIMM(),
        'p_loss': 5000,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.01,
        'g_high': 0.5,
        'ri_min': 1,
        'ri_max': 100,
    }
)

register(
    id='revisit-v02',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=1e3),
        'p_loss': 5000,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 1,
        'ri_max': 250,
        'is_pomdp': True
    }
)

register(
    id='revisit-v03',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': DefinedKalman(),
        'p_loss': 10,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.01,
        'g_high': 0.3,
        'ri_min': 1,
        'ri_max': 250,
        'is_pomdp': False
    }
)

register(
    id='revisit-v04',
    entry_point='radargym.single_target_tracking:RevisitIntervalBenchmarkDiscrete',
    kwargs={
        'sims': [
                    BaselineKalman(
                        n_max=20,
                        var=(4.5 * 9.81) ** 2,
                        traj_idx=idx,
                        P0=None,
                        beamwidth=0.02,
                        pfa=1e-6,
                        sn0=50) for idx in range(6)],
        'p_loss': 100,
        'ri_min': 1,
        'ri_max': 250,
        'n_act': 10,
        'n_discretize': 10,
        'g_low': 0.5,
        'g_high': 1.25
    }
)

register(
    id='revisit-v05',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=(4.5 * 9.81) ** 2, theta_accuracy=0.001),
        'p_loss': 100,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 10,
        'ri_max': 250,
        'noisy_obs': True
    }
)

register(
    id='revisit-v06',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=(4.5 * 9.81) ** 2, theta_accuracy=0.002),
        'p_loss': 100,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 10,
        'ri_max': 250,
        'noisy_obs': True
    }
)

register(
    id='revisit-v07',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=(4.5 * 9.81) ** 2, theta_accuracy=0.01),
        'p_loss': 100,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 10,
        'ri_max': 250,
        'noisy_obs': True
    }
)

register(
    id='revisit-v08',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=(4.5 * 9.81) ** 2, theta_accuracy=0.02),
        'p_loss': 100,
        'n_discretize': 10,
        'n_act': 10,
        'g_low': 0.5,
        'g_high': 1.25,
        'ri_min': 10,
        'ri_max': 250,
        'noisy_obs': True
    }
)


register(
    id='revisit-v10',
    entry_point='radargym.single_target_tracking:RevisitIntervalContinuous',
    kwargs={
        'sim': BaselineKalman(traj_idx=2, var=1e3),
        'p_loss': 5000,
        'n_act': 10,
        'ri_min': 1,
        'ri_max': 100,
    }
)


register(
    id='revisit-v30',
    entry_point='radargym.single_target_tracking:MMRevisitIntervalDiscrete',
    kwargs={
        'sim': DefinedCVCAIMM(),
        'p_loss': 10,
        'n_discretize': 4,
        'n_act': 20,
        'g_low': 0.2,
        'g_high': 0.8,
        'ri_min': 1,
        'ri_max': 100,
    }
)

register(
    id='revisit-v20',
    entry_point='radargym.single_target_tracking:RevisitIntervalBenchmarkDiscrete',
    kwargs={
        'sims': [IMMBenchmark(traj_idx=idx) for idx in range(6)],
        'p_loss': 100,
        'ri_min': 1,
        'ri_max': 75,
        'n_act': 10,
        'n_discretize': 10,
        'g_low': 0.05,
        'g_high': 0.5
    }
)

register(
    id='revisit-v21',
    entry_point='radargym.single_target_tracking:MMRevisitIntervalBenchmarkDiscrete',
    kwargs={
        'sims': [IMMBenchmark(traj_idx=idx) for idx in range(6)],
        'p_loss': 100,
        'ri_min': 1,
        'ri_max': 75,
        'n_act': 10,
        'n_discretize': 10,
        'g_low': 0.05,
        'g_high': 0.95
    }
)

register(
    id='revisit-v22',
    entry_point='radargym.single_target_tracking:RevisitIntervalBenchmarkDiscrete',
    kwargs={
        'sims': [IMMBenchmark(traj_idx=idx) for idx in range(6)],
        'p_loss': 100,
        'ri_min': 1,
        'ri_max': 75,
        'n_act': 10,
        'n_discretize': 10,
        'g_low': 0.05,
        'g_high': 0.5,
        'multi_dim_obs': True
    }
)

register(
    id='revisit-v23',
    entry_point='radargym.single_target_tracking:MMRevisitIntervalBenchmarkDiscrete',
    kwargs={
        'sims': [IMMBenchmark(traj_idx=idx) for idx in range(6)],
        'p_loss': 100,
        'ri_min': 1,
        'ri_max': 75,
        'n_act': 10,
        'n_discretize': 10,
        'g_low': 0.05,
        'g_high': 0.95,
        'multi_dim_obs': True
    }
)
