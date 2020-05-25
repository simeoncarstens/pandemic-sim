# pandemic-sim
Boredom and curiosity hit and I programmed a little pandemic simulator. It is heavily inspired by (one could also say, ripped off from) this one here: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/ 
It works as follows: people wander around in a room and upon contact can infect each other. Adjustable features are:

- probability to catch the disease
- probability to spread the disease
- base probability for infection depending on distance between persons
- time it takes for people to become healthy again after infection
- probability of a person to die in a certain time step, given they are infected
- people fancily bounce off each other and off walls, courtesy of numerical solution of equations of motions and repulsive force between people, whose parameters can be set

## Requirements
You'll need Python 3.6 or greater, `matplotlib` and `numpy`. The most current `matplotlib` version (3.2.1) has a bug which might prevent the rendering result from being saved, so you want to have an older version. You also need `ffmpeg`. You can install all necessary Python packages (including a compatible `matplotlib` version) by running
```
$ pip install -r requirements.txt
```
If you're using [Nix](https://nixos.org) or you'd like to try it out (and you should!), you can just use the provided `shell.nix` to get an environment with all necessary dependencies, `ffmpeg` included.

## Warning
This is just a toy. I'm not an epidemiologist, you're probably neither, so don't use results of this simulation for anything other than superficial illustration purposes.
