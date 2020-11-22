"""
Classes for numerically simulating an epidemic.
"""

from abc import abstractmethod, ABCMeta

import numpy as np


class Person(object):
    def __init__(self, pos, vel, personal_disease_model,
                 personal_transmission_model=None,
                 infected=False, immune=False):
        """
        Make a new person object.

        Args:
            pos (np.ndarray): The x/y position vector of the person.
            vel (np.ndarray): The x/y velocity vector of the person.
            personal_disease_model: disease model which calculates death
                                    and cure probabilities
            personal_transmission_model: Individual part of disease model
                                         modeling transmission of disease
            infected (bool): Whether the person is infected or not
            immune (bool): Whether person is immune or not
        """
        self.pos = pos
        self.vel = vel
        self.infected = infected
        self.immune = immune
        self.personal_disease_model = personal_disease_model
        self.personal_transmission_model = personal_transmission_model
        self.infected_since = None
        self._dead = False

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
    def __init__(self, geometry, persons, health_system, transmission_model,
                 particle_engine):
        """
        Arguments:
        
        - geometry (Geometry): A geometry object which defines the space in
                               which persons are moving
        - persons (list): A list of persons
        - health_system (HealthSystem): A HealthSystem object which possibly
                                        influences the death probability
        - transmission_model (AbstractTransmissionModel): Disease model which 
                                                          models transmission.
        - particle_engine: A ParticleEngine object which is responsible for the
                           physical part of the simulation
        """
        self.geometry = geometry
        self._check_for_personal_models(persons)
        self.persons = persons
        self.health_system = health_system
        self.particle_engine = particle_engine
        self._transmission_model = transmission_model
        self.particle_engine.skip_condition = lambda i: self.persons[i].dead

        def pairwise_hook(i, j, d):
            return self._transmission_model.maybe_transmit(persons[i], persons[j], d)

        self.particle_engine.pairwise_hook = pairwise_hook
        self._current_step = 0
        self._update_population_state()

    @property
    def population_state(self):
        return self._population_state.copy()
        
    def _check_for_personal_models(self, persons):
        if any((p.personal_transmission_model is None for p in persons)):
            raise ValueError(("All persons need the 'personal_transmission_model' "
                              "attribute set"))
        if any((p.personal_disease_model is None for p in persons)):
            raise ValueError(("All persons need the 'personal_disease_model' "
                              "attribute set"))
        
    def _move_persons(self):
        """
        Update movement of all persons during one time step
        """
        old_poss = np.array([p.pos for p in self.persons])
        old_vels = np.array([p.vel for p in self.persons])
        new_poss, new_vels = self.particle_engine.integration_step(old_poss,
                                                                   old_vels)
        for (p, pos, vel) in zip(self.persons, new_poss, new_vels):
            p.pos = pos
            # sanitize velocities of dead people: the old gradient in the
            # integrator does not get updated when a person dies, so they
            # will have a non-zero velocity 
            p.vel = vel if not p.dead else np.array([0, 0])

    def _update_population_state(self):
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

        self._population_state = {'n_dead': n_dead, 'n_infected': n_infected,
                                  'n_alive': n_alive}


    def _possibly_kill_person(self, p, population_state):
        """
        Kills a person with a certain probability based on the population
        macrostate. Returns a  boolean indicating whether the person has been
        killed or not.
        """
        hs_death_prob = self.health_system.calculate_death_probability_factor(
            p, self._current_step, population_state)
        personal_death_prob = (
            p.personal_disease_model.calculate_death_probability_factor(
                self._current_step))
        if np.random.random() < hs_death_prob * personal_death_prob:
            p.dead = True
            return True
        else:
            return False


    def _possibly_cure_person(self, p):
        """
        Possibly cures a person based on the time they have been infected and
        the time it takes to heal and become immune. Returns a boolean
        indicating whether the person has been cured or not.
        """
        personal_cure_prob = (
            p.personal_disease_model.calculate_cure_probability_factor(
                self._current_step))
        hs_cure_prob = self.health_system.calculate_cure_probability_factor(
            p, self._current_step, self.population_state)
        
        if np.random.random() < personal_cure_prob * hs_cure_prob:
            p.infected = False
            p.immune = True
            return True
        else:
            return False


    def _evolve_person_disease_states(self):
        """
        Evolves the disease status of all persons. That currently comprises
        the random events of death and being cured.
        """
        for p in self.persons:
            if p.infected:
                if p.infected_since is None:
                    # person is freshly infected, thus set time stamp
                    p.infected_since = self._current_step
                else:
                    # person has been infected in previous simulation step
                    person_cured = self._possibly_cure_person(p)
                    if not person_cured:
                        self._possibly_kill_person(p, self.population_state)
        self._update_population_state()


    def step(self):
        """
        Advance the simulation by one time step.
        """
        self._move_persons()
        self._evolve_person_disease_states()
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

