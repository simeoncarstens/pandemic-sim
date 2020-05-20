# pandemic-sim
Boredom and curiosity hit and I programmed a little pandemic simulator. It is heavily inspired by (one could also say, ripped off from) this one here: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/ 
It works as follows: people wander around in a room and upon contact can infect each other. Adjustable features are:

- probability to catch the disease
- probability to spread the disease
- base probability for infection depending on distance between persons
- time it takes for people to become healthy again after infection
- probability of a person to die in a certain time step, given they are infected
- people fancily bounce off each other and off walls, courtesy of numerical solution of equations of motions and repulsive force between people, whose parameters can be set

## Warning
This is just a toy. I'm not an epidemiologist, you're probably neither, so don't use results of this simulation for anything other than superficial illustration purposes.
