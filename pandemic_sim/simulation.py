"""
Classes for numerically simulating an epidemic.
"""

from abc import abstractmethod, ABCMeta

import numpy as np


class Person(object):
    def __init__(self, pos, vel, in_prob, out_prob, death_prob, infected=False,
                 immune=False):
        """
        Make a new person object.

        Args:
            pos (np.ndarray): The x/y position vector of the person.
            vel (np.ndarray): The x/y velocity vector of the person.
            in_prob (float): Probability to get infected in an
                             encounter with an infected person.
            out_prob (float): Probability to infect another person in
                              an encounter.
            death_prob (float): Probability of dying in a single time
                                step if person is infected.
            infected (bool): Whether the person is infected or not
            immune (bool): Whether person is immune or not
        """
        self.pos = pos
        self.vel = vel
        self.in_prob = in_prob
        self.out_prob = out_prob
        self.infected = infected
        self.immune = immune
        self.infected_since = None
        self._dead = False
        self.death_prob = death_prob

    @property
    def dead(self):
        return self._dead
    @dead.setter
    def dead(self, value):
        """
        Sets a person's status to dead. This involves setting
        their infection status to False and their velocity to zero.
        """
        if self.dead and value == False:
            raise("Zombie mode not implemented yet")
        elif ~self.dead and value == True:
            self._dead = True
            self.infected = False
            self.vel[:] = 0.0


class Geometry(metaclass=ABCMeta):
    def __init__(self, force_constant):
        """
        A class defining the geometry of the space the persons move in.

        Arguments:

        - force_constant (float): Force constant determining strength of 
                                  the interaction potential between a 
                                  person and the walls
                                  The higher it is, the stronger a person
                                  bounces off a wall.
        """
        self.force_constant = force_constant


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
    def __init__(self, width, height, force_constant):
        """
        A rectangular geometry.

        Arguments:
        
        - width (float): the width of the rectangle
        - height (float): the height of the rectangle
        """
        self.width = width
        self.height = height
        
        super().__init__(force_constant)


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

        return self.force_constant * res


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
    

class Simulation(object):
    def __init__(self, geometry, persons, health_system, prob_dist, dt,
                 cutoff, transmit_cutoff, force_constant, time_to_heal):
        """
        Arguments:
        
        - geometry (Geometry): A geometry object which defines the space in
                               which persons are moving
        - persons (list): A list of persons
        - health_system (HealthSystem): A HealthSystem object which possibly
                                        influences the death probability
        - prob_dist (callable): A function describing a base probability of
                                infection depending on the distance between
                                two persons
        - dt (float): Time step for numerical integration of equations of 
                      motion
        - cutoff (float): Cutoff distance between two persons below which
                          the force pushing them away from each other kicks
                          in
        - transmit_cutoff (float): Cutoff distance between two persons above
                                   which no transmission can happen
        - force_constant (float): Force constant determining strength of 
                                  the interaction potential between two persons.
                                  The higher it is, the more persons bounce
                                  off each other.
        - time_to_heal (int): Number of time steps it takes for a person to
                              become healthy again after they've been infected
        """
        self.geometry = geometry
        self.persons = persons
        self.health_system = health_system
        self.prob_dist = prob_dist
        self.dt = dt
        self.cutoff = cutoff
        self.transmit_cutoff = transmit_cutoff
        self.force_constant = force_constant
        poss = np.array([p.pos for p in self.persons])
        self._oldgrad = self.inter_person_gradient(poss) + self.geometry.gradient(poss)
        self.time_to_heal = time_to_heal
        self._current_step = 0
        
    
    def _maybe_transmit(self, p1, p2, d):
        """
        Transmits disease between two persons depending on their distance
        
        Arguments:

        - p1 (Person): first person
        - p2 (Person): second person
        - d (float): distance between the two persons
        """
        if (p1.infected & (~p2.infected)) or ((~p1.infected) & p2.infected):
            base_prob = self.prob_dist(d)
            if p1.infected & ~p2.immune:
                total_prob = base_prob * p1.out_prob * p2.in_prob
                p2.infected = np.random.random() < total_prob
            elif p2.infected & ~p1.immune:
                total_prob = base_prob * p2.out_prob * p1.in_prob
                p1.infected = np.random.random() < total_prob


    def inter_person_gradient(self, pos):
        """
        Gradient of potential energy responsable for making persons
        bounce off each other. Includes call to _maybe_transmit
        in order to avoid a second double loop.

        Arguments:
        
        - pos (np.ndarray): 2D numpy array of position vectors of
                            all persons

        Returns:
        - a numpy.ndarrray containing the gradient due to interactions
          between persons
        """
        dm = np.linalg.norm(pos[None, :] - pos[:, None], axis=2)
        res = np.zeros(pos.shape)
        for i, pos1 in enumerate(pos):
            if self.persons[i].dead:
                ## dead persons don't interact with others
                continue
            for j, pos2 in enumerate(pos[i+1:], i+1):
                if self.persons[j].dead:
                    continue
                if dm[i,j] < self.transmit_cutoff:
                    self._maybe_transmit(self.persons[i], self.persons[j], dm[i,j])
                if dm[i,j] < self.cutoff:
                    f = (self.cutoff - dm[i,j]) / dm[i,j] * (pos1 - pos2)
                    res[i] += f
                    res[j] -= f
                    
        return self.force_constant * res


    def _integration_step(self, poss, vels):
        """
        One single velocity Verlet integration step

        Arguments:

        - poss (np.ndarray): 2D numpy array of position vectors
        - vels (np.ndarray): 2D numpy array of velocity vectors

        Returns:

        - (poss, vels, new_grad): Updated positions, velocities, and gradient
        """
        poss = poss.copy()
        vels = vels.copy()
        poss += vels * self.dt + 0.5 * self._oldgrad * self.dt * self.dt
        new_grad = self.inter_person_gradient(poss) + self.geometry.gradient(poss)
        vels += 0.5 * (self._oldgrad + new_grad) * self.dt

        return poss, vels, new_grad
    
    
    def _move_persons(self):
        """
        Update movement of all persons during one time step
        """
        old_poss = np.array([p.pos for p in self.persons])
        old_vels = np.array([p.vel for p in self.persons])
        new_poss, new_vels, new_grad = self._integration_step(old_poss,
                                                              old_vels)

        ## we cache the gradient
        self._oldgrad = new_grad

        for i, (pos, vel) in enumerate(zip(new_poss, new_vels)):
            self.persons[i].pos = pos
            self.persons[i].vel = vel
        

    def _calculate_population_state(self):
        """
        Calculates overall quantities characterizing the state of the population:
        number of dead people, number of infected people, number of alive people

        Returns:

        - a dictionary with the macroscopic population state. Keys are
          "n_dead", "n_infected", "n_alive".
        """
        microstates = np.array([(p.dead, p.infected) for p in self.persons])
        n_dead = microstates[:,0].sum()
        n_infected = microstates[:,1].sum()
        n_alive = len(self.persons) - n_dead

        return {'n_dead': n_dead, 'n_infected': n_infected, 'n_alive': n_alive}
           
            
    def step(self):
        """
        Advance the simulation by one time step.
        """
        self._move_persons()

        population_state = self._calculate_population_state()
        for p in self.persons:
            if p.infected:
                if p.infected_since is None:
                    p.infected_since = self._current_step
                elif self._current_step - p.infected_since >= self.time_to_heal:
                    p.infected = False
                    p.immune = True
                else:
                    death_prob = self.health_system.calculate_death_probability_factor(
                        population_state)
                    death_prob *= p.death_prob
                    if np.random.random() < death_prob:
                        p.dead = True
        self._current_step += 1
                    

    def run(self, n_steps):
        all_positions = [np.array([p.pos for p in self.persons])]
        all_infected = [np.array([True if p.infected else False
                                  for i, p in enumerate(self.persons)])]
        all_immune = [np.array([p.immune for p in self.persons])]
        all_fatalities = [np.array([p.dead for p in self.persons])]

        for i in range(n_steps):
            if i % 50 == 0:
                print(f"Simulating step {i}/{n_steps}...")
            self.step()

            ## keep track of all positions, infection status, immunity
            ## status and dead persons
            poss = np.array([p.pos for p in self.persons])
            all_positions.append(poss)
            infected = np.array([p.infected for p in self.persons])
            all_infected.append(infected)
            immune = np.array([p.immune for p in self.persons])
            all_immune.append(immune)
            fatalities = np.array([p.dead for p in self.persons])
            all_fatalities.append(fatalities)
            
        all_positions = np.array(all_positions)
        all_infected = np.array(all_infected)
        all_immune = np.array(all_immune)
        all_fatalities = np.array(all_fatalities)
        all_healthy = ~all_infected & ~all_fatalities

        print("Done.")
        
        return {'all_positions': all_positions, 'all_healthy': all_healthy,
                'all_infected': all_infected, 'all_immune': all_immune,
                'all_fatalities': all_fatalities}

