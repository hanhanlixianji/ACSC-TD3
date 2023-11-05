from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla.envs.carla_env:CarlaEnv',
)

register(
    id='carla_test_muti-v0',
    entry_point='gym_carla.envs.env_test_muti:CarlaEnv',
)

register(
    id='carla_test_only-v0',
    entry_point='gym_carla.envs.env_test_only:CarlaEnv',
)

register(
    id='notrick_env_only-v0',
    entry_point='gym_carla.envs.notrick_env_only:CarlaEnv',
)

register(
    id='env_safe_pure_only-v0',
    entry_point='gym_carla.envs.env_safe_pure_only:CarlaEnv',
)

register(
    id='scenario1_safe-v0',
    entry_point='gym_carla.envs.scenario1_safe:CarlaEnv',
)