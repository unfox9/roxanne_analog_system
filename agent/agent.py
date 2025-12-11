from abc import ABC, abstractmethod


class Agent(ABC):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""
