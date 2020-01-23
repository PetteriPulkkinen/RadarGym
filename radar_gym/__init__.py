from gym.envs.registration import register


register(
    id='Tracking-v0',
    entry_point='radar_gym.single_target_tracking:LinearTrackingEnv',
)
