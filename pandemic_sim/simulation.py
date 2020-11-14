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


class Simulation(object):
    def __init__(self, geometry, persons, health_system, prob_dist,
                 max_transmit_distance, particle_engine, time_to_heal):
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
        - max_transmit_distance (float): Maximum distance between two persons
                                         above which no transmission can happen
        - particle_engine: A ParticleEngine object which is responsible for the
                           physical part of the simulation
        - time_to_heal (int): Number of time steps it takes for a person to
                              become healthy again after they've been infected
        """
        self.geometry = geometry
        self.persons = persons
        self.health_system = health_system
        self.prob_dist = prob_dist
        self.particle_engine = particle_engine
        self.particle_engine.skip_condition = lambda i: self.persons[i].dead
        self.particle_engine.pairwise_hook = self._maybe_transmit
        self.max_transmit_distance = max_transmit_distance
        self.time_to_heal = time_to_heal
        self._current_step = 0
        
    
    def _maybe_transmit(self, p1_index, p2_index, d):
        """
        Transmits disease between two persons depending on their distance
        
        Arguments:

        - p1 (Person): first person
        - p2 (Person): second person
        - d (float): distance between the two persons
        """
        p1 = self.persons[p1_index]
        p2 = self.persons[p2_index]
        if (p1.infected & (~p2.infected)) or ((~p1.infected) & p2.infected):
            base_prob = self.prob_dist(d)
            if p1.infected & ~p2.immune:
                total_prob = base_prob * p1.out_prob * p2.in_prob
                p2.infected = np.random.random() < total_prob
            elif p2.infected & ~p1.immune:
                total_prob = base_prob * p2.out_prob * p1.in_prob
                p1.infected = np.random.random() < total_prob


    def _move_persons(self):
        """
        Update movement of all persons during one time step
        """
        old_poss = np.array([p.pos for p in self.persons])
        old_vels = np.array([p.vel for p in self.persons])
        new_poss, new_vels = self.particle_engine.integration_step(old_poss,
                                                                   old_vels)
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

