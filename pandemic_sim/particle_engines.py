"""
Particle engines and numerical integrators - all that's needed to simulate a
physical system.
"""

from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractIntegrator(metaclass=ABCMeta):
    def __init__(self, timestep, gradient):
        """
        Defines the interface for numerical integrators.

        Arguments:

        - timestep (float): Integration timestep
        - gradient (callable): Gradient of the potential energy
                               field particles move in
        """
        self.timestep = timestep
        self.gradient = gradient

    @abstractmethod
    def step(self, poss, vels):
        """
        Simulates a single time step.

        Arguments:
        
        - poss (np.ndarray): 2D numpy array of position vectors
        - vels (np.ndarray): 2D numpy array of velocity vectors
        
        """
        pass


class VelocityVerletIntegrator(AbstractIntegrator):
    def __init__(self, timestep, gradient):
        """
        Velocity Verlet integration scheme.
        
        Arguments:

        - timestep (float): Time step for numerical integration of equations
                            of motion

        """
        super().__init__(timestep, gradient)
        self._oldgrad = None
        
    def step(self, poss, vels):
        """
        One single velocity Verlet integration step

        Arguments:

        - poss (np.ndarray): 2D numpy array of position vectors
        - vels (np.ndarray): 2D numpy array of velocity vectors

        Returns:

        - (poss, vels, new_grad): Updated positions and velocities
        """
        poss = poss.copy()
        vels = vels.copy()
        if self._oldgrad is None:
            self._oldgrad = self.gradient(poss)
        poss += self.timestep * (vels + 0.5 * self._oldgrad * self.timestep)
        new_grad = self.gradient(poss)
        vels += 0.5 * (self._oldgrad + new_grad) * self.timestep
        self._oldgrad = new_grad
        
        return poss, vels        


class AbstractParticleEngine(metaclass=ABCMeta):
    def __init__(self, cutoff, integrator_params,
                 integrator_class=VelocityVerletIntegrator,
                 skip_condition=None, pairwise_hook=None):
        """
        Defines the basic interface for particle engines which defince and
        simulate a physical system by integration of equations of motion.
        Currently assumes that particle-particle interactions are only
        pairwise and limited to inter-particle distances below a certain
        cutoff.

        Arguments:

        - cutoff (float): Distance between two particles above which no
                          interaction occurs
        - integrator_params (dict): Dictionary of parameters the Integrator
                                    object requires for instantiation.
        - integrator_class (AbstractIntegrator): An integrator class which,
                                                 once instantiated, is
                                                 responsible for numerical
                                                 integration of the particles'
                                                 equations of motion.
        - skip_condition (callable): A function taking a a single argument
                                     (the index of a particle) and which,
                                     when returning True, will lead to skipping
                                     the calculation of the particles' gradient,
                                     setting it to zero.
        - pairwise_hook (callable): A function taking three parameters (two
                                    particle indices and the particles'
                                    distance). Allows to avoid additional nested
                                    loops for pairwise stuff other than
                                    gradient calculation.
        """
        self.cutoff = cutoff
        self.integrator = self._make_integrator(integrator_class,
                                                **integrator_params)
        self.skip_condition = skip_condition or (lambda _: False)
        self.pairwise_hook = pairwise_hook or (lambda *args: None)
        self._params = {}
        self._gradient_list = []

    def _make_integrator(self, integrator_class, **integrator_params):
        """
        Factory method instantiating the integrator object.

        Arguments:

        - integrator_class (AbstractIntegrator): Integrator class to be
                                                 instantiated
        - integrator_params (dict): Any parameters the integrator might take
        """
        return integrator_class(gradient=self.gradient, **integrator_params)

    def gradient(self, poss):
        """
        Total gradient acting on particles.

        Arguments:

        - poss (numpy.ndarray): Positions of particles
        """
        return np.sum([gradient(poss) for gradient in self._gradient_list], 0)

    def integration_step(self, poss, vels):
        """
        Integrates the equations of motions for one time step.

        Arguments:
        
        - poss (numpy.ndarray): positions of the particles
        - vels (numpy.ndarray): velocities of the particles
        """
        return self.integrator.step(poss, vels)


class DefaultParticleEngine(AbstractParticleEngine):
    def __init__(self, cutoff, geometry_gradient, integrator_params,
                 integrator_class=VelocityVerletIntegrator,
                 inter_particle_force_constant=20.0,
                 geometry_force_constant=20.0, skip_condition=None,
                 pairwise_hook=None):
        """
        Default particle engine which simulates a physical system with
        repulsive pairwise forces and repulsive forces exerted by the
        geometry the particles are confined in.

        Arguments:

        - cutoff (float): Distance between two particles above which no
                          interaction occurs.
        - geometry_gradient (callable): A function calculating the gradient
                                        due to the repulsive force of the
                                        geometry on the particles.
        - integrator_params (dict): Dictionary of parameters the Integrator
                                    object requires for instantiation.
        - integrator_class (AbstractIntegrator): An integrator class which,
                                                 once instantiated, is
                                                 responsible for numerical
                                                 integration of the particles'
                                                 equations of motion.
        - inter_particle_force_constant (float): Force constant determining the
                                                 strength of repulsive forces
                                                 acting between particles
        - geometry_force_constant (float): Force constant determining the
                                           strength of the force confining
                                           particles to the geometry.                                           
        - skip_condition (callable): A function taking a a single argument
                                     (the index of a particle) and which,
                                     when returning True, will lead to skipping
                                     the calculation of the particles' gradient,
                                     setting it to zero.
        - pairwise_hook (callable): A function taking three parameters (two
                                    particle indices and the particles'
                                    distance). Allows to avoid additional nested
                                    loops for pairwise stuff other than
                                    gradient calculation.
        """
        super().__init__(cutoff, integrator_params, integrator_class,
                         skip_condition, pairwise_hook)
        self.inter_particle_force_constant = inter_particle_force_constant
        self.geometry_force_constant = geometry_force_constant
        self._gradient_list.append(self.inter_particle_gradient)
        self._gradient_list.append(geometry_gradient)
        self._oldgrad = None
        
    def inter_particle_gradient(self, pos):
        """
        Gradient of potential energy responsable for making particles
        bounce off each other. Includes call to _maybe_transmit
        in order to avoid a second double loop.

        Arguments:
        
        - pos (np.ndarray): 2D numpy array of position vectors of
                            all particles

        Returns:
        - a numpy.ndarrray containing the gradient due to interactions
          between particles
        """
        dm = np.linalg.norm(pos[None, :] - pos[:, None], axis=2)
        res = np.zeros(pos.shape)
        for i, pos1 in enumerate(pos):
            if self.skip_condition(i):
                continue
            for j, pos2 in enumerate(pos[i+1:], i+1):
                if self.skip_condition(j):
                    continue
                self.pairwise_hook(i, j, dm[i,j])
                if dm[i,j] < self.cutoff:
                    f = (self.cutoff - dm[i,j]) / dm[i,j] * (pos1 - pos2)
                    res[i] += f
                    res[j] -= f
                    
        return self.inter_particle_force_constant * res
