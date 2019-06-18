import numpy as np
import copy

class OUNoise:
    """
    Ornstein-Uhlenbeck process.

    add some noise to our actions, in order to encourage exploratory behavior.
    And since our actions translate to force and torque being applied to a
    quadcopter, we want consecutive actions to not vary wildly.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
