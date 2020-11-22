"""
Models for dying or getting cured from the disease
"""

from abc import ABCMeta, abstractmethod


class AbstractPersonalDiseaseModel(metaclass=ABCMeta):
    def __init__(self, person):
        """
        Defines the interface for individual disease models which determine
        probabilitites for persons dying or getting cured during a simulation
        time step.

        Arguments:

        - person (Person): the person affected by the disease described by this
                           model.
        """
        self.person = person
        
    @abstractmethod
    def calculate_death_probability_factor(self, time):
        """
        Calculates the probability factor telling how likely the person is to
        die from the disease in a given time step. Might depend on the
        simulation time.

        Arguments:

        - time (int): current simulation time
        """
        pass

    @abstractmethod
    def calculate_cure_probability_factor(self, time):
        """
        Calculates the probability factor telling how likely the person is to
        be cured from the disease in a given time step. Might depend on the
        simulation time.

        Arguments:

        - time (int): current simulation time
        """
        pass


class DefaultPersonalDiseaseModel(AbstractPersonalDiseaseModel):
    def __init__(self, person, death_probability, time_to_heal):
        """
        - time_to_heal (int): Number of time steps it takes for a person to
                              become healthy again after they've been infected

        """
        super().__init__(person)
        self.death_probability = death_probability
        self.time_to_heal = time_to_heal

    def calculate_cure_probability_factor(self, time):
        """
        Returns one if person has been infected for some time larger than the
        time a person needs to heal, else zero.

        Arguments:

        - time (int): current simulation time
        """
        if time - self.person.infected_since > self.time_to_heal:
            return 1.0
        else:
            return 0.0

    def calculate_death_probability_factor(self, _):
        """
        Returns a constant probability to die within a time step.
        """
        return self.death_probability
