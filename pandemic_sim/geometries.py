"""
Classes defining geoemtries particles move around in
"""

from abc import ABCMeta, abstractmethod

import numpy as np


class Geometry(metaclass=ABCMeta):
    def __init__(self):
        """
        A class defining the geometry of the space the persons move in.
        """


    @abstractmethod
    def gradient(self, pos):
        """
        The gradient acting on persons due to the geometry.

        Returns:

        - a numpy.ndarray containing the gradient due to the walls
        """
        pass


    @abstractmethod
    def get_random_position(self):
        """
        Get a random position within the boundaries of the geometry

        Returns:
        
        - a numpy.ndarray with random positions valid in the geometry
        """
        pass


    @abstractmethod
    def get_random_position_set(self, n_positions):
        """
        Gets a set of positions with ideally on average high distances between
        each position to avoid overlaps.

        Arguments:

        - n_positions (int): number of random positions to draw
        """
        pass

    
class RectangleGeometry(Geometry):
    def __init__(self, width, height):
        """
        A rectangular geometry.

        Arguments:
        
        - width (float): the width of the rectangle
        - height (float): the height of the rectangle
        """
        super().__init__()
        self.width = width
        self.height = height


    def gradient(self, pos):
        """
        The gradient acting on a person due to a collision with the four walls
        of this geometry.

        Arguments:

        - pos (numpy.ndarray): 2D position vector of a person

        Returns:

        - a numpy.ndarray containing the gradient
        """
        res = np.zeros(pos.shape)

        ## upper wall
        cond = pos[:,1] > self.height
        res[cond, 1] -= pos[cond, 1] - self.height
        ## lower wall
        cond = pos[:,1] < 0.0
        res[cond, 1] -= pos[cond, 1]
        ## right wall
        cond = pos[:,0] > self.width
        res[cond, 0] -= pos[cond, 0] - self.width
        ## left wall
        cond = pos[:,0] < 0.0
        res[cond, 0] -= pos[cond, 0]

        return res


    def get_random_position(self):
        """
        Get a random position within the four walls defined by this geometry

        Returns:

        - a numpy.ndarray with random positions
        """
        return np.random.uniform(low=(0,0), high=(self.width, self.height))

    
    def get_random_position_set(self, n_positions):
        """
        Gets a set of positions with ideally on average high distances between
        each position to avoid overlaps.

        Arguments:

        - n_positions (int): number of random positions to draw
        """
        w = self.width
        h = self.height
        n_x = int(np.ceil(w * np.sqrt(n_positions) / np.sqrt(w * h)))
        n_y = int(np.ceil(h * np.sqrt(n_positions) / np.sqrt(w * h)))
        all_coords = [(x, y) for x in np.linspace(0, self.width, n_x)
                      for y in np.linspace(0, self.height, n_y)]
        all_coords = np.array(all_coords)
        return all_coords[np.random.choice(np.arange(len(all_coords)),
                                           n_positions, False)]
