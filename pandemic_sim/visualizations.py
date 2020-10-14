from abc import abstractmethod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Visualization:
    def __init__(self, simulation_results):
        self._simulation_results = simulation_results


    @property
    def figure(self):
        return self._figure
        
        
    @abstractmethod
    def _setup_figure(self):
        pass
    
        
    @abstractmethod
    def visualize_single_step(self, step):
        pass


class GeometryDrawer(object):
    def __init__(self, geometry):
        self.geometry = geometry
        
        
    @abstractmethod
    def draw(self, ax):
        pass
    

class RectangleGeometryDrawer(GeometryDrawer):
    def draw(self, ax, radius):
        ax.set_xlim((0, self.geometry.width + radius))
        ax.set_ylim((0, self.geometry.height + radius))


class DefaultVisualization(Visualization):
    def __init__(self, simulation_results, geometry_drawer, radius):
        super(DefaultVisualization, self).__init__(simulation_results)
        self._geometry_drawer = geometry_drawer
        self._radius = radius
        
        self._check_result_shapes()
        positions = self._simulation_results['all_positions']
        self._n_steps = positions.shape[0]
        self._n_persons = positions.shape[1]

        self._setup_figure()
        self._setup_axes()

        
    def _check_result_shapes(self):
        positions = self._simulation_results['all_positions']
        infected = self._simulation_results['all_infected']
        healthy = self._simulation_results['all_healthy']
        immune = self._simulation_results['all_immune']
        fatalities = self._simulation_results['all_fatalities']
        if not len(positions) == len(infected) == len(healthy) \
           == len(fatalities) == len(immune):
            raise ValueError("Simulation result arrays have unequal lengths")


    def _setup_figure(self):
        self._figure = plt.figure()
    

    def _setup_axes(self):
        gs = gridspec.GridSpec(3, 4)

        main_ax = self._figure.add_subplot(gs[:,:3])
        self._geometry_drawer.draw(main_ax, self._radius)
        main_ax.set_aspect('equal')
        main_ax.set_xticks(())
        main_ax.set_yticks(())

        infected_ax = self._figure.add_subplot(gs[0,3])
        infected_ax.set_xlim((0, self._n_steps))
        infected_ax.set_ylim((0, self._n_persons))

        immune_ax = self._figure.add_subplot(gs[1,3])
        immune_ax.set_xlim((0, self._n_steps))
        immune_ax.set_ylim((0, self._n_persons))

        fatalities_ax = self._figure.add_subplot(gs[2,3])
        fatalities_ax.set_xlim((0, self._n_steps))
        fatalities_ax.set_ylim((0, self._n_persons // 3))

        self._main_ax = main_ax
        self._infected_ax = infected_ax
        self._immune_ax = immune_ax
        self._fatalities_ax = fatalities_ax


    def _plot_infected(self, currently_infected):
        inf_circles = [plt.Circle(pos, radius=self._radius, linewidth=0)
                       for pos in currently_infected]
        inf_c = matplotlib.collections.PatchCollection(inf_circles,
                                                       color="orange")
        self._main_ax.add_collection(inf_c)


    def _plot_infected_time(self, time_series):
        self._infected_ax.plot(time_series, color='black')
        self._infected_ax.set_xlabel('time')
        self._infected_ax.set_ylabel('# infected')


    def _plot_healthy(self, currently_healthy):
        healthy_circles = [plt.Circle(pos, radius=self._radius, linewidth=0)
                           for pos in currently_healthy]
        healthy_c = matplotlib.collections.PatchCollection(healthy_circles,
                                                           color="blue")
        self._main_ax.add_collection(healthy_c)


    def _plot_dead(self, currently_dead):
        dead_circles = [plt.Circle(pos, radius=self._radius, linewidth=1)
                        for pos in currently_dead]
        dead_c = matplotlib.collections.PatchCollection(dead_circles,
                                                        color="lightgray")
        self._main_ax.add_collection(dead_c)


    def _plot_fatalities_time(self, time_series):
        self._fatalities_ax.plot(time_series, color='black')
        self._fatalities_ax.set_xlabel('time')
        self._fatalities_ax.set_ylabel('# fatalities')


    def _plot_immune_time(self, time_series):
        self._immune_ax.plot(time_series, color='black')
        self._immune_ax.set_xlabel('time')
        self._immune_ax.set_ylabel('# immune')


    def visualize_single_step(self, step):
        positions = self._simulation_results['all_positions']
        infected = self._simulation_results['all_infected']
        immune = self._simulation_results['all_immune']
        fatalities = self._simulation_results['all_fatalities']
        healthy = self._simulation_results['all_healthy']

        currently_infected = positions[step, infected[step]]
        currently_healthy = positions[step, healthy[step]]
        currently_dead = positions[step, fatalities[step]]
            
        self._plot_infected(currently_infected)
        self._plot_infected_time(infected[:step].sum(1))
        self._plot_healthy(currently_healthy)
        self._plot_immune_time(immune[:step].sum(1))
        self._plot_dead(currently_dead)
        self._plot_fatalities_time(fatalities[:step].sum(1))

        self.figure.tight_layout()
