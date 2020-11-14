import numpy as np

from pandemic_sim.simulation import Person, Simulation
from pandemic_sim.geometries import RectangleGeometry
from pandemic_sim.health_systems import SimpleHealthSystem
from pandemic_sim.particle_engines import (DefaultParticleEngine,
                                           VelocityVerletIntegrator)
from pandemic_sim.visualizations import (DefaultVisualization,
                                         DefaultPersonsDrawer,
                                         RectangleGeometryDrawer,
                                         SimpleHealthSystemCurvesPlotter)
from pandemic_sim.animators import CelluloidAnimator


## Initialize simulation
n_persons = 200
room = RectangleGeometry(25, 25, force_constant=20.0)

# base probability of transmission during one timestep of an encounter between
# two persons
base_prob = 0.00015

initial_positions = room.get_random_position_set(n_persons)
max_vel = 2
persons = [Person(pos,
                  np.random.uniform(low=(-max_vel, -max_vel),
                                    high=(max_vel, max_vel), size=2),
                  1.0, 1.0,
                  base_prob, False)
           for pos in initial_positions]
# some persons actually start out being infected
chosen_ones = np.random.choice(np.arange(n_persons), n_persons // 50)
for i in chosen_ones:
    persons[i].infected = True

# Simulate a health system which can get overwhelmed if the number of infected
# persons rises above a certain threshold. In that case, the death probability
# is increased fivefold
health_system = SimpleHealthSystem(threshold=50, death_probability_factor=5.0)
    
# Set up the particle engine responsible for physical simulation
pe = DefaultParticleEngine(cutoff=0.75, geometry_gradient=room.gradient,
                           integrator_params={'timestep': 0.1},
                           integrator_class=VelocityVerletIntegrator,
                           inter_particle_force_constant=20.0,
                           geometry_force_constant=20.0)

# Set up simulation object
sim = Simulation(room, persons, health_system, lambda d:  d < 1,
                 max_transmit_distance=3, particle_engine=pe,
                 time_to_heal=150)
n_steps = 300
sim_result = sim.run(n_steps)

radius = pe.cutoff / 2
curves_plotter = SimpleHealthSystemCurvesPlotter(health_system)
viz = DefaultVisualization(sim_result, RectangleGeometryDrawer(room),    
                           DefaultPersonsDrawer(radius),
                           curves_plotter)

animator = CelluloidAnimator(viz)
# interval=2 means that only every second step is shown in the video,
# decreasing the time it takes for the animation to be created
animator.animate(n_steps, interval=2)
