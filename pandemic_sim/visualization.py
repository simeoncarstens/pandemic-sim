from abc import abstractmethod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from celluloid import Camera


def plot_infected(currently_infected, radius, ax):
    inf_circles = [plt.Circle(pos, radius=radius, linewidth=0)
                   for pos in currently_infected]
    inf_c = matplotlib.collections.PatchCollection(inf_circles,
                                                   color="orange")
    ax.add_collection(inf_c)


def plot_infected_time(time_series, ax):
    ax.plot(time_series, color='black')
    ax.set_xlabel('time')
    ax.set_ylabel('# infected')


def plot_healthy(currently_healthy, radius, ax):
    healthy_circles = [plt.Circle(pos, radius=radius, linewidth=0)
                       for pos in currently_healthy]
    healthy_c = matplotlib.collections.PatchCollection(healthy_circles,
                                                       color="blue")
    ax.add_collection(healthy_c)
    

def plot_dead(currently_dead, radius, ax):
    dead_circles = [plt.Circle(pos, radius=radius, linewidth=1)
                    for pos in currently_dead]
    dead_c = matplotlib.collections.PatchCollection(dead_circles,
                                                    color="lightgray")
    ax.add_collection(dead_c)


def plot_fatalities_time(time_series, ax):
    ax.plot(time_series, color='black')
    ax.set_xlabel('time')
    ax.set_ylabel('# fatalities')


def plot_immune_time(currently_immune, ax):
    ax.plot(currently_immune, color='black')
    ax.set_xlabel('time')
    ax.set_ylabel('# immune')


def setup_axes(fig, room, n_steps, n_persons):
    gs = gridspec.GridSpec(3, 4)
    
    main_ax = fig.add_subplot(gs[:,:3])
    main_ax.set_xlim((0, room.w))
    main_ax.set_ylim((0, room.h))
    main_ax.set_aspect('equal')
    main_ax.set_xticks(())
    main_ax.set_yticks(())

    inf_ax = fig.add_subplot(gs[0,3])
    inf_ax.set_xlim((0, n_steps))
    inf_ax.set_ylim((0, n_persons))

    immune_ax = fig.add_subplot(gs[1,3])
    immune_ax.set_xlim((0, n_steps))
    immune_ax.set_ylim((0, n_persons))

    fatalities_ax = fig.add_subplot(gs[2,3])
    fatalities_ax.set_xlim((0, n_steps))
    fatalities_ax.set_ylim((0, n_persons // 3))

    return main_ax, inf_ax, immune_ax, fatalities_ax


class Animator:
    @abstractmethod
    def __init__(self, all_positions, all_infected, all_healthy, all_immune,
                 all_fatalities, room, radius):
        self._all_positions = all_positions
        self._all_infected = all_infected
        self._all_healthy = all_healthy
        self._all_immune = all_immune
        self._all_fatalities = all_fatalities
        self._room = room
        self._radius = radius
        if len(all_positions) == len(all_infected) == len(all_healthy) \
           == len(all_fatalities):
            self._n_steps = len(all_positions)
            self._n_persons = self._all_positions.shape[1]
        else:
            raise ValueError("Simulation result arrays have unequal lengths")
        self._init_fig_axes()


    def _init_fig_axes(self):
        self._fig = plt.figure()
        self._main_ax, self._infected_ax, self._immune_ax, self._fatalities_ax = \
            setup_axes(self._fig, self._room, self._n_steps, self._n_persons)


    @abstractmethod
    def visualize_single_step(self, step):
        pass
        

    @abstractmethod
    def animate(self):
        pass


class CelluloidAnimator(Animator):
    def __init__(self, all_positions, all_infected, all_healthy, all_fatalities,
                 all_immune, room, radius, frame_rate=10, out="output.mp4"):
        super(CelluloidAnimator, self).__init__(all_positions, all_infected,
                                                all_healthy, all_immune,
                                                all_fatalities, room, radius)
        self._frame_rate = frame_rate
        self._out = out


    def visualize_single_step(self, step):

        poss = self._all_positions[step]
        currently_infected = self._all_positions[step, self._all_infected[step]]
        currently_healthy = self._all_positions[step, self._all_healthy[step]]
        currently_immune = self._all_positions[step, self._all_immune[step]]
        currently_dead = self._all_positions[step, self._all_fatalities[step]]
            
        plot_infected(currently_infected, self._radius, self._main_ax)
        plot_infected_time(self._all_infected[:step].sum(1),
                           self._infected_ax)
        plot_healthy(currently_healthy, self._radius, self._main_ax)
        plot_immune_time(self._all_immune[:step].sum(1),
                         self._immune_ax)
        plot_dead(currently_dead, self._radius, self._main_ax)
        plot_fatalities_time(self._all_fatalities[:step].sum(1),
                             self._fatalities_ax)
        
        
    def animate(self):
        camera = Camera(self._fig)
        for step in range(self._n_steps):
            if step % 50 == 0:
                print(f"Animating step {step}/{self._n_steps}...")
            self.visualize_single_step(step)
            
            self._fig.tight_layout()
            camera.snap()

        anim = camera.animate(blit=True)
        anim.save(self._out, fps=self._frame_rate)
