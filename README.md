[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/simeoncarstens/pandemic-sim/master?filepath=simulation.ipynb)
# Agent-based (?) simulation of an epidemic
Boredom and curiosity hit and I programmed a little something to simulate the outbreak of an infectious disease. It is heavily inspired by (one could also say, ripped off from) this one here: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/  
It works as follows: people wander around in a room and upon contact can infect each other. Adjustable settings are as follows.
### Environmental factors
- size / shape of the room persons move around in
- treatment capacity of the health system (increased death probability when number of infections threshold is exceeded)

### Persons
- number of persons
- probability to catch the disease
- probability to expose others
- time it takes for people to become healthy again after infection
- probability of a person to die in a certain simulation time step, given they are infected
- minimal distance people maintain to each other
- initial infection status

### Disease
- base probability for infection depending on distance between persons
- time it takes to heal once a person is infected
- maximum distance below which the disease can be transmitted

### Numerical parameters
- integration time step
- strength of repulsive forces pushing persons away from each other and from walls

## Visualization and animation
In the default visualization, for a given simulation step, the room is drawn with persons as points, colored by their status: healthy (blue), infected (orange), dead (gray). Furthermore, the numbers of currently infected and immune persons as well as the cumulative number of fatalities is plotted.
Here's how that looks like:

![Oh, the simulated horror...](http://simeon-carstens.com/files/sim_example2.png)

Currently, there is a default animation class, which uses `ffmpeg` to write out a `.mp4` video showing the evolution of the epidemic.

## Installation and requirements

You can easily try all this out yourself without installing anything by clicking on the Binder button at the top of this readme file! It will launch an interactive Jupyter notebook, in which parameters are explained. You can change them to your liking and check the result, all without installing anything on your machine.

### Requirements
If you want to run this code locally, you'll need Python 3.6 or greater, `matplotlib` and `numpy`. The most current `matplotlib` version (3.2.1) has a bug which might prevent the rendering result from being saved, so you want to have an older version. You also need `ffmpeg`. You can install all necessary Python packages (including a compatible `matplotlib` version) by running
```
$ pip install -r requirements.txt
```

### Nix shell
If you're using [Nix](https://nixos.org) or you'd like to try it out (and you should!), you can just use the provided `shell.nix` to get an environment with all necessary dependencies, `ffmpeg` included.

## Special thanks
...to

- [Tru Huynh](https://github.com/truatpasteurdotfr) for the Binder setup

## Warning
This is just a toy. I'm not an epidemiologist (proof? I didn't know _exactly_ what a pandemic is until _after_ I named this repository), you're probably neither, so don't use results of this simulation for anything other than superficial illustration purposes.
