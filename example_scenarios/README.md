# Example scenarios
This directory contains a few example scenarios:
- `no_measures.py`: a reference scenario in which persons move around a lot and no effect of masks is simulated.
- `masks_ony.py`: identical to `no_measures.py`, only that the probably for persons to expose others is severely decreased, simulating the effect of wearing a simple face mask which reduces others' exposition, but doesn't protect from incoming viral particles
- `masks_low_mobility.py`: identical to `masks_only.py`, only that additionally, the velocities with which the persons move around is significantly lowered.

All other parameters (number of persons, room size, time it takes to heal, health system capacity etc.) assume equal values for all scenarios.

While these simulations are stochastic due to the random initial states, you should see, at least on average, the following resulting fatalities:
- `no_measures.py`: ~25 to 30 fatalities
- `masks_only.py`: ~20 fatalities
- `masks_low_mobility.py`: at most a few fatalities
