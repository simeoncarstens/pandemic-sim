"""
Models for disease transmission
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np


class AbstractPersonalTransmissionModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, person):
        """
        Defines the interface for the person-specific part of a transmission
        model.

        Arguments:

        - person (Person): The person this model instance applies to.
        """
        pass

    def calculate_exposure_results(self):
        """
        Returns a namedtuple with data pertaining to the exposure probability.
        """
        return self.ExposureResult(self.out_prob)

    def calculate_susceptibility_results(self):
        """
        Returns a namedtuple with data pertaining to the probability for being
        susceptible.
        """
        return self.SusceptibilityResult(self.in_prob)


class DefaultPersonalTransmissionModel(AbstractPersonalTransmissionModel):
    ExposureResult = namedtuple('DefaultExposureResult', 'probability')
    SusceptibilityResult = namedtuple('DefaultSusceptibilityResult',
                                      'probability')
    
    def __init__(self, person, in_prob, out_prob):
        """
        Person-specific part of a transmission model based on probabilities to
        get infected and to expose others (modeling wearing of masks / other
        barrier measures).

        Arguments:

        - person (Person): The person this model instance applies to.
        - in_prob (float): Probability to get infected if exposed.
        - out_prob (float): Probability to expose another person if person this
                            object relates to is infected.
        """
        super().__init__(person)
        self.in_prob = in_prob
        self.out_prob = out_prob

    def calculate_exposure_results(self):
        """
        Returns a namedtuple with the exposure probability.
        """
        return self.ExposureResult(self.out_prob)

    def calculate_susceptibility_results(self):
        """
        Returns a namedtuple with a probability for being susceptible.
        """
        return self.SusceptibilityResult(self.in_prob)

    
class AbstractPairwiseTransmissionModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        """
        Defines the interface for the pairwise part of a transmission model.
        """
        pass

    @abstractmethod
    def calculate_pairwise_results(self, person1, person2, **kwargs):
        """
        Calculates the part of the transmission probability depending on the
        distance between two persons.
        
        Arguments:

        - person1 (Person): First person.
        - person2 (Person): Second person.
        - kwargs (dict): Possibly additional arguments such as the
                         pre-calculated distance between two persons.

        """
        pass


class DefaultPairwiseTransmissionModel(AbstractPairwiseTransmissionModel):
    Result = namedtuple('PairwiseResult', 'probability')
    
    def __init__(self, prob_dist):
        """
        Pairwise part of a transmission model in which transmission probability
        depends on the distance between two persons.

        Arguments:

        - prob_dist (callable): A function taking the distance between two
                                persons and returning a probability.
        """
        super().__init__()
        self.prob_dist = prob_dist

    def calculate_pairwise_results(self, _, __, distance):
        """
        Calculates the part of the transmission probability depending on the
        distance between two persons.
        
        Arguments:

        - distance (float): Distance between two persons.
        """
        return self.Result(self.prob_dist(distance))


class AbstractTransmissionModel(metaclass=ABCMeta):
    def __init__(self, global_tm):
        """
        Defines the interface for transmission models modeling the mechanism of
        disease transmission between two persons in a spatially close
        encounter.

        Arguments:

        - pairwise_tm (AbstractPairwiseTransmissionMondel): Pairwise part of
                                                            the transmission
                                                            model.
        """
        self.global_tm = global_tm

    @abstractmethod
    def personal_tm_factory(self, person, **kwargs):
        """
        Factory method which produces the appropriate personal transmission
        models.

        Arguments:

        - person (Person): The person the which to-be-produced transmission
                           model object relates.
        - kwargs (dict): Arguments the personal transmission model constructor
                         needs.
        """
        pass
        
    def maybe_transmit(self, person1, person2, distance):
        """
        Transmits disease between two persons depending on their distance
        
        Arguments:

        - person1 (Person): first person
        - person2 (Person): second person
        - distance (float): distance between the two persons
        """
        if not (person1.immune and person2.immune
                or person1.infected and person2.infected
                or person1.dead or person2.dead):
            pairwise_results = self.global_tm.calculate_pairwise_results(
                person1, person2, distance)
            p1_tm = person1.personal_transmission_model
            p2_tm = person2.personal_transmission_model
            if person1.infected:
                person2.infected = self._combine_results(
                    pairwise_results,
                    p1_tm.calculate_exposure_results(),
                    p2_tm.calculate_susceptibility_results())
            else:
                person1.infected = self._combine_results(
                    pairwise_results,
                    p2_tm.calculate_exposure_results(),
                    p1_tm.calculate_susceptibility_results())

    @abstractmethod
    def _combine_results(self, pairwise_results, exposure_result,
                         susceptibility_result):
        """
        Combines pairwise and per-person results to calculate the new
        infection status of both persons.
        
        Arguments:

        - pairwise_results (namedtuple): Named tuple with pairwise-property-
                                         dependent results influencing the 
                                         transmission.
        - exposure_results (namedtuple): Named tuple with person-specific
                                         results influencing the exposure.
        - susceptibility_results (namedtuple): Named tuple with person-specific
                                               results influencing the
                                               susceptibility.
        """
        pass


class DefaultTransmissionModel(AbstractTransmissionModel):
    def __init__(self, prob_dist):
        """
        Defines the interface for transmission models modeling the mechanism of
        disease transmission between two persons in a spatially close encounter.

        Arguments:

        - prob_dist (callable): A function taking the distance between two
                                persons and returning a probability.
        """
        super().__init__(DefaultPairwiseTransmissionModel(prob_dist))

    @staticmethod
    def personal_tm_factory(person, in_prob, out_prob):
        """
        Factory method which produces a DefaultPersonalTransmissionModel.

        Arguments:

        - person (Person): The person the which to-be-produced transmission
                           model object relates.
        - in_prob (float): Part of the transmission probability quantifying
                           susceptibility of the person.
        - out_prob (float): Part of the transmission probability quantifying
                            how like a person is to expose another.

        """
        return DefaultPersonalTransmissionModel(person, in_prob, out_prob)
    
    def _combine_results(self, pairwise_results, exposure_results,
                         susceptibility_results):
        """
        Transmits disease between two persons randomly depending on their
        distance and on individual probabilities for being infected and
        exposing another person.
        
        Arguments:

        - pairwise_results (namedtuple): Named tuple with distance-based part
                                         of total transmission probability.
        - exposure_results (namedtuple): Named tuple with person-specific
                                        probability for exposing another person.
        - susceptibility_results (namedtuple): Named tuple with person-specific
                                               probability for being infected
                                               when exposed.
        """
        pw_prob = pairwise_results.probability
        infection_prob = pw_prob * exposure_results.probability \
            * susceptibility_results.probability

        return np.random.random() < infection_prob
