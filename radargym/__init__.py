from gym.envs.registration import register
from trackingsimpy.simulation.revisit_interval import DefinedIMM, BaselineKalman


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
        'sim': BaselineKalman(traj_idx=4, var=1e3),
        'p_loss': 5000,
        'n_obs': 10,
        'n_act': 10,
        'g_low': 0.1,
        'g_high': 1.0,
        'ri_min': 1,
        'ri_max': 100,
    }
)

register(
    id='revisit-v01',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': DefinedIMM(),
        'p_loss': 5000,
        'n_obs': 10,
        'n_act': 10,
        'g_low': 0.1,
        'g_high': 1.0,
        'ri_min': 1,
        'ri_max': 250,
    }
)

register(
    id='revisit-v02',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete',
    kwargs={
        'sim': BaselineKalman(traj_idx=4, var=1e3),
        'p_loss': 5000,
        'n_obs': 10,
        'n_act': 10,
        'g_low': 0.1,
        'g_high': 1.0,
        'ri_min': 1,
        'ri_max': 100,
        'is_pomdp': True
    }
)

register(
    id='revisit-v10',
    entry_point='radargym.single_target_tracking:RevisitIntervalContinuous',
    kwargs={
        'sim': BaselineKalman(traj_idx=4, var=1e3),
        'p_loss': 5000,
        'n_act': 10,
        'ri_min': 1,
        'ri_max': 100,
    }
)