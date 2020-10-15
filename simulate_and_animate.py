import numpy as np

from pandemic_sim.simulation import (Person,
                                     Simulation,
                                     RectangleGeometry,
                                     SimpleHealthSystem)
from pandemic_sim.visualizations import (DefaultVisualization,
                                         RectangleGeometryDrawer,
                                         DefaultPersonsDrawer,
                                         SimpleHealthSystemCurvesPlotter)
from pandemic_sim.animators import CelluloidAnimator


## Initialize simulation
n_persons = 500
room = RectangleGeometry(40, 40, force_constant=20.0)
# persons start with random positions and velocities
persons = [Person(room.get_random_position(),
                  np.random.uniform(low=(-2, -2), high=(2, 2), size=2),
                  0.2, 0.2, 0.00015, False)
           for _ in range(n_persons)]
# some persons actually start out being infected
chosen_ones = np.random.choice(np.arange(n_persons), n_persons // 50)
for i in chosen_ones:
    persons[i].infected = True

health_system = SimpleHealthSystem(threshold=150, death_probability_factor=5.0)
    
day_unit = 10
sim = Simulation(room, persons, health_system, lambda d:  d < 1,
                 dt=0.1,
                 cutoff=0.75,
                 transmit_cutoff=3,
                 force_constant=20,
                 time_to_heal=day_unit * 14)
n_steps = 50 * day_unit
sim_result = sim.run(n_steps)

radius = sim.cutoff / 2
curves_plotter = SimpleHealthSystemCurvesPlotter(health_system)
viz = DefaultVisualization(sim_result, RectangleGeometryDrawer(room),    
                           DefaultPersonsDrawer(radius),
                           curves_plotter)
animator = CelluloidAnimator(viz)
animator.animate(n_steps)
