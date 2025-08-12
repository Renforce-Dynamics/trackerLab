import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="G1TrackWalkRun",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_track_cfg:G1TrackWalkRun",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:G1TrackWalkRun",
    },
)
