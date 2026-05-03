from custom_envs import SimpleRewardEnv

env = SimpleRewardEnv(
    gun="m4_carbine",
    n_targets=5, target_radius=5.0,
    wind_mean=(3.0, 0.0, 0.0), wind_gust_sigma=1.0,
    recoil_noise=0.04, recoil_angle_noise_deg=1.0,
    casings_enabled=False, bullets_enabled=False,
    control_level="setpoint",   # or "thrust"
    seed=42,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(reward)
