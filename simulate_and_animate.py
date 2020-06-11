import numpy as np

from pandemic_sim.simulation import Person, Simulation, Room
from pandemic_sim.visualizations import DefaultVisualization
from pandemic_sim.visualizations import BonusVisualization
from pandemic_sim.animators import CelluloidAnimator


## Initialize simulation
day_unit = 10
def get_random_velocity(max_vel):
        return np.random.uniform(low=-max_vel, high=max_vel, size=2)
    
## Case 1: basically no distancing - everyone gets it
if True:
    n_persons = 500
    room = Room(40, 40)
    in_prob = 0.2
    out_prob = 0.2
    death_prob = 0.00015
    infected_frac = 50
    tth = 14
    n_steps = day_unit * 50
    max_vel = 2
    cutoff = 0.75
    Visualization = DefaultVisualization
    
## Case 2: restricting movements - outbreaks limited to clusters
if not True:
    n_persons = 500
    room = Room(40, 40)
    in_prob = 0.2
    out_prob = 0.2
    death_prob = 0.00015
    infected_frac = 50
    tth = 14
    max_vel = 0.2
    n_steps = day_unit * 50
    cutoff = 0.75
    Visualization = DefaultVisualization
    
## Case 3: full movement, but increased physical distancing
if not True:
    n_persons = 500
    room = Room(40, 40)
    in_prob = 0.2
    out_prob = 0.2
    death_prob = 0.00015
    infected_frac = 50
    tth = 14
    n_steps = day_unit * 10
    max_vel = 2
    cutoff = 1.25
    Visualization = DefaultVisualization


## Case 4: mysterious epidemic in the Paris office
if True:
    n_persons = 20
    room = Room(15, 15)
    in_prob = 1.0
    out_prob = 1.0
    death_prob = 0.0
    infected_frac = 20
    tth = 14000000
    n_steps = day_unit * 30
    max_vel = 2
    cutoff = 0.75
    Visualization = BonusVisualization


## get non-overlapping initial positions
print("Finding non-overlapping initial positions...", end="")
def get_random_position():
    return np.random.uniform(low=(0,0), high=(room.w, room.h))
poss = np.empty((n_persons, 2))
for i in range(n_persons):
    if i == 0:
        poss[i] = get_random_position()
    else:
        while True:
            pos = get_random_position()
            if np.all(np.linalg.norm(pos[None,:] - poss[:i], axis=1) > 0.75):
                poss[i] = pos
                break
print(" Done.")
persons = [Person(poss[i],
                  get_random_velocity(max_vel),
                  in_prob, out_prob, death_prob, False)
           for i in range(n_persons)]
# some persons actually start out being infected
chosen_ones = np.random.choice(np.arange(n_persons), n_persons // infected_frac)
for i in chosen_ones:
    persons[i].infected = True


sim = Simulation(room, persons, lambda d:  d < 1,
                 dt=0.1,
                 cutoff=cutoff,
                 transmit_cutoff=1,
                 force_constant=20,
                 time_to_heal=day_unit * tth)
sim_result = sim.run(n_steps)


viz = Visualization(sim_result, room, 0.75 / 2)
animator = CelluloidAnimator(viz)
animator.animate(n_steps)
