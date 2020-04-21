from gym.envs.registration import register

register(
    id='Tracking-v0',
    entry_point='radargym.single_target_tracking:LinearTrackingDiscrete'
)

register(
    id='Tracking-v1',
    entry_point='radargym.single_target_tracking:LinearTrackingSmooth'
)

register(
    id='revisit-v0',
    entry_point='radargym.single_target_tracking:RevisitIntervalDiscrete'
)

register(
    id='revisit-v1',
    entry_point='radargym.single_target_tracking:RevisitIntervalContinuous'
)
