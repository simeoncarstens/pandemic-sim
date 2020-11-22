import numpy as np

from pandemic_sim.simulation import Person, Simulation
from pandemic_sim.geometries import RectangleGeometry
from pandemic_sim.health_systems import SimpleHealthSystem
from pandemic_sim.particle_engines import (DefaultParticleEngine,
                                           VelocityVerletIntegrator)
from pandemic_sim.disease_models import DefaultPersonalDiseaseModel
from pandemic_sim.transmission_models import DefaultTransmissionModel
from pandemic_sim.visualizations import (DefaultVisualization,
                                         DefaultPersonsDrawer,
                                         RectangleGeometryDrawer,
                                         SimpleHealthSystemCurvesPlotter)
from pandemic_sim.animators import CelluloidAnimator


## Initialize simulation
n_persons = 200
room = RectangleGeometry(25, 25)

# Set up transmission model, which models how the disease is transmitted
# upon an encounter of two persons
transmission_model = DefaultTransmissionModel(lambda d: d < 1)

# probability of dying during a timestep
death_prob = 0.00015
# duration of a person's infection (in units of simulation time steps)
time_to_heal = 150

initial_positions = room.get_random_position_set(n_persons)
max_vel = 2
persons = [Person(pos,
                  np.random.uniform(low=(-max_vel, -max_vel),
                                    high=(max_vel, max_vel), size=2),
                  personal_disease_model=None,
                  personal_transmission_model=None, immune=False)
           for pos in initial_positions]
# Add individual parts of transmission model to persons
for p in persons:
    p.personal_transmission_model = transmission_model.personal_tm_factory(
        p, in_prob=1.0, out_prob=1.0)
    p.personal_disease_model = DefaultPersonalDiseaseModel(p, death_prob,
                                                           time_to_heal)
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
sim = Simulation(room, persons, health_system, transmission_model,
                 particle_engine=pe)
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
