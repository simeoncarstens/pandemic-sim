import numpy as np

from pandemic_sim.simulation import (Person,
                                     Simulation,
                                     RectangleGeometry,
                                     SimpleHealthSystem)
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
# reduce outgoing probability by 95%; meaning that a person is less likely to
# spread the virus in its environment. This emulates efficient filtering of
# breathed-out air.
out_prob = 0.05
# do not reduce ingoing probability, meaning that a person catches the virus
# with the base probability if theyare close to a person having spread
# the virus. This emulates no filtering of inhaled air by masks
in_prob = 1.0
# significantly decrease mobility of persons as compared to the "only_masks.py"
# scenario
max_vel = 0.5
initial_positions = room.get_random_position_set(n_persons)
persons = [Person(pos,
                  np.random.uniform(low=(-max_vel, -max_vel),
                                    high=(max_vel, max_vel), size=2),
                  in_prob, out_prob,
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
    
sim = Simulation(room, persons, health_system, lambda d:  d < 1,
                 dt=0.1,
                 cutoff=0.75,
                 transmit_cutoff=3,
                 force_constant=20,
                 time_to_heal=150)
n_steps = 800
sim_result = sim.run(n_steps)

radius = sim.cutoff / 2
curves_plotter = SimpleHealthSystemCurvesPlotter(health_system)
viz = DefaultVisualization(sim_result, RectangleGeometryDrawer(room),    
                           DefaultPersonsDrawer(radius),
                           curves_plotter)

animator = CelluloidAnimator(viz)
# interval=4 means that only every sixth step is shown in the video,
# decreasing the time it takes for the animation to be created
animator.animate(n_steps, interval=6)
