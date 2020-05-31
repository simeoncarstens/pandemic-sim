import numpy as np

from pandemic_sim.simulation import Person, Simulation, Room
from pandemic_sim.visualization import CelluloidAnimator


## Initialize simulation
n_persons = 500
room = Room(40, 40)
# persons start with random positions and velocities
persons = [Person(np.random.uniform(low=(0,0), high=(room.w, room.h)),
                  np.random.uniform(low=(-2, -2), high=(2, 2), size=2),
                  0.2, 0.2, 0.00015, False)
           for _ in range(n_persons)]
# some persons actually start out being infected
chosen_ones = np.random.choice(np.arange(n_persons), n_persons // 50)
for i in chosen_ones:
    persons[i].infected = True


day_unit = 10
sim = Simulation(room, persons, lambda d:  d < 1,
                 dt=0.1,
                 cutoff=0.75,
                 transmit_cutoff=3,
                 force_constant=20,
                 time_to_heal=day_unit * 14)
sim_result = sim.run(50 * day_unit)

animator = CelluloidAnimator(**sim_result, radius=sim.cutoff / 2,
                             room=room)
animator.animate()
