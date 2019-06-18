import numpy as np
from physics_sim import PhysicsSim

# takeoff, hover in place, land softly, or reach a target pose.

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self) -> float:
        """Uses current pose of sim to return reward."""

        # average difference
        # reward:float = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # TODO see torcs environment

        new_reward:float = 0.

        current_dims = self.sim.pose[:3]
        cx, cy, cz = tuple(current_dims)

        target_dims = self.target_pos
        tx, ty, tz = tuple(target_dims)

        z_reward:float = 0.
        x_reward:float = 0.
        y_reward:float = 0.
        dims_w: tuple = (0., 0., 0.)

        # prioritize z dimension first
        if (abs(cz - tz) / 100) in np.arange(0, 0.05, 0.01): # until we're within striking distance of z...
            dims_w = (2, .5, .5)
        else:
            dims_w = (.9, 1.05, 1.05)

        z_w, x_w, y_w = (dims_w)
        z_reward = z_w * abs(cz - tz)
        x_reward = x_w * abs(cx - tx)
        y_reward = y_w * abs(cy - ty)

        new_reward = 1. - .3 * (z_reward + x_reward + y_reward)

        # return reward
        return new_reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
