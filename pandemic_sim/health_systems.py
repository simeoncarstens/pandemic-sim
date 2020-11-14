"""
Classes defining "health systems", meaning exterior influences on the
simulation conditioned on the simulation macro variables like the number of
infected people.
"""

from abc import ABCMeta, abstractmethod


class HealthSystem(metaclass=ABCMeta):
    @abstractmethod
    def calculate_death_probability_factor(self, population_state):
        """
        Calculates a factor which depends on the populaton macrostate
        (number of infected persons, number of total persons) and by
        which in a simulation the death probability is multiplied

        Arguments:

        - population_state (dict): dictionary of quantities defining the overall
                                   state of the population (number of infected
                                   persons, number of total persons)
        """
        pass


class NoEffectHealthSystem(HealthSystem):
    def calculate_death_probability_factor(self, _):
        """
        Returns a factor of 1.0 indepent from the population state.
        """

        return 1.0

    
class SimpleHealthSystem(HealthSystem):
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

        
    def calculate_death_probability_factor(self, population_state):
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
