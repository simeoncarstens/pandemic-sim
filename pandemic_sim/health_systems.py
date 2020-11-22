"""
Classes defining "health systems", meaning exterior influences on the
simulation conditioned on the simulation macro variables like the number of
infected people.
"""

from abc import ABCMeta, abstractmethod


class AbstractHealthSystem(metaclass=ABCMeta):
    @abstractmethod
    def calculate_death_probability_factor(self, population_state):
        """
        In general, the probability of a person dying during one simulaton
        step might depend on that person, the current simulation time, and
        the population state.

        Arguments:

        - p (Person): the person the cure probability is calculated for
        - time (int): the current simulation time
        - population_state (dict): dictionary of quantities defining the overall
                                   state of the population (number of infected
                                   persons, number of total persons)
        """
        pass

    @abstractmethod
    def calculate_cure_probability_factor(self, p, time, population_state):
        """
        In general, the probability of a person being cured during one 
        simulation step might depend on that person, the current simulation
        time, and the population state.

        Arguments:

        - p (Person): the person the cure probability is calculated for
        - time (int): the current simulation time
        - population_state (dict): dictionary of quantities defining the overall
                                   state of the population (number of infected
                                   persons, number of total persons)
        """
        pass


class NoEffectHealthSystem(AbstractHealthSystem):
    def calculate_death_probability_factor(self, _, __, ___):
        """
        This health system doesn't influence the death probability, so this
        always returns 1.0
        """
        return 1.0

    def calculate_cure_probability_factor(self, _, __, ___):
        """
        This health system doesn't influence the cure probability, so this
        returns 1.0
        """
        return 1.0

    
class SimpleHealthSystem(AbstractHealthSystem):
    def __init__(self, threshold, death_probability_factor):
        """
        Simple health system which simulates limited capacity by increasing
        the death probability if the number of infected people exceeds a 
        certain threshold

        Arguments:
        
        - thresold (int): critical number of infected persons
        - death_probability_factor (float): factor with which the default death
                                            probability gets multiplied in case
                                            of too many infected persons
        """
        self.threshold = threshold
        self.death_probability_factor = death_probability_factor

        
    def calculate_death_probability_factor(self, _, __, population_state):
        """
        Returns the death probability factor if number of infected people
        exceeds a certain threshold, else returns 1.0

        Arguments:

        - population_state (dict): dictionary of quantities defining the overall
                                   state of the population (number of infected
                                   persons, number of total persons)
        """
        
        n_infected = population_state['n_infected']
        if n_infected > self.threshold:
            return self.death_probability_factor
        else:
            return 1.0

    def calculate_cure_probability_factor(self, _, __, ___):
        """
        This health system doesn't influence the cure probability, so this
        returns 1.0
        """
        return 1.0
