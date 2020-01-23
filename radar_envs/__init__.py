from gym.envs.registration import register


register(
    id='Tracking-v0',
    entry_point='radar_envs.single_target_tracking:LinearTrackingEnv',
)
