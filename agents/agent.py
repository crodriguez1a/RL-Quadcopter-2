from .ddpg import DDPG

# tweak various hyperparameters and the reward function

class MyAgent(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
