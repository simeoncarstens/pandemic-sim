"""
Different visualizations of an epidemic simulation
"""
from abc import abstractmethod, ABCMeta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GeometryDrawer(metaclass=ABCMeta):
    def __init__(self, geometry):
        """
        Abstract class defining the interface for geometry drawers. They are
        responsible for drawing the room / geometry the persons move in.

        Arguments:
        
        - geometry (Geometry): a geometry object defining the room geometry
        """
        self.geometry = geometry
        
        
    @abstractmethod
    def draw(self, ax):
        """
        Draws the geometry in a matplotlib axis.

        Arguments:

        - ax (axis): matplotlib axis to draw geometry in
        """
        pass
    

class RectangleGeometryDrawer(GeometryDrawer):
    def draw(self, ax):
        """
        Draws a rectangular geometry in a matplotlib axis.

        Arguments:

        - ax (axis): matplotlib axis to draw geometry in
        """
        ax.set_xlim((0, self.geometry.width))
        ax.set_ylim((0, self.geometry.height))


class PersonsDrawer(metaclass=ABCMeta):
    def __init__(self, radius):
        """
        Abstract class defining the interface for persons drawers. They are
        responsible for drawing persons as cirles.

        Arguments:

        - radius (float): the radius of a circle representing a single person
        """
        self._radius = radius


    @abstractmethod
    def draw(self, current_simulation_state, ax):
        """
        Draws circular persons in a matplotlib axis.

        Arguments:

        - current_simulation_state (dict): the current simulation state
                                           containing numpy arrays with
                                           positions for infected and dead 
                                           persons.
        - ax (axis): matplotlib axis to draw persons in
        """
        pass


class DefaultPersonsDrawer(PersonsDrawer):
    def __init__(self, radius):
        """
        This default drawer draws three kind of circular persons, each in a
        different color: healthy (blue), infected (orange), and dead (gray).

        Arguments:

        - radius (float): the radius of a circle representing a single person
        """
        super().__init__(radius)


    def draw(self, current_simulation_state, ax):
        """
        Draws healthy, infected and dead persons as circles in a matplotlib 
        axis.

        Arguments:

        - current_simulation_state (dict): the current simulation state
                                           containing numpy arrays with
                                           positions for infected and dead 
                                           persons.
        - ax (axis): matplotlib axis to draw persons in
        """
        self._draw_infected(current_simulation_state['infected'], ax)
        self._draw_healthy(current_simulation_state['healthy'], ax)
        self._draw_dead(current_simulation_state['dead'], ax)


    def _draw_infected(self, currently_infected, ax):
        """
        Draws currently infected persons as orange circles.

        Arguments:

        - currently_infected (numpy.ndarray): positions of infected persons
        - ax (axis): matplotlib axis to draw the persons in
        """
        inf_circles = [plt.Circle(pos, radius=self._radius, linewidth=0)
                       for pos in currently_infected]
        inf_c = matplotlib.collections.PatchCollection(inf_circles,
                                                       color="orange")
        ax.add_collection(inf_c)
        

    def _draw_healthy(self, currently_healthy, ax):
        """
        Draws currently healthy persons as blue circles.

        Arguments:

        - currently_healthy (numpy.ndarray): positions of healthy persons
        - ax (axis): matplotlib axis to draw the persons in
        """
        healthy_circles = [plt.Circle(pos, radius=self._radius, linewidth=0)
                           for pos in currently_healthy]
        healthy_c = matplotlib.collections.PatchCollection(healthy_circles,
                                                           color="blue")
        ax.add_collection(healthy_c)


    def _draw_dead(self, currently_dead, ax):
        """
        Draws currently dead persons as gray circles.

        Arguments:

        - currently_dead (numpy.ndarray): positions of dead persons
        - ax (axis): matplotlib axis to draw the persons in
        """
        dead_circles = [plt.Circle(pos, radius=self._radius, linewidth=1)
                        for pos in currently_dead]
        dead_c = matplotlib.collections.PatchCollection(dead_circles,
                                                        color="lightgray")
        ax.add_collection(dead_c)


class CurvesPlotter(metaclass=ABCMeta):
    def __init__(self):
        """
        Abstract class defining the interface for curve plotters. They are
        responsible for plotting the simulation macrostate (number of infected,
        immune and dead persons).
        """
        pass

    
    @abstractmethod
    def plot_infected_time(self, time_series, ax):
        """
        Plots time number of infected people over simulation time.

        Arguments:

        - time_series (numpy.ndarray): time series of number of infected
                                       persons
        - ax (axis): matplotlib axis to draw time series in
        """
        pass
    

    @abstractmethod
    def plot_immune_time(self, time_series, ax):
        """
        Plots time number of immune people over simulation time.

        Arguments:

        - time_series (numpy.ndarray): time series of number of immune
                                       persons
        - ax (axis): matplotlib axis to draw time series in
        """
        pass


    @abstractmethod
    def plot_fatalities_time(self, time_series, ax):
        """
        Plots time number of fatalities over simulation time.

        Arguments:

        - time_series (numpy.ndarray): time series of number of 
                                       fatalities
        - ax (axis): matplotlib axis to draw time series in
        """
        pass


class DefaultCurvesPlotter(CurvesPlotter):
    def plot_infected_time(self, time_series, ax):
        """
        Plots time number of infected people over simulation time as a simple
        black line.

        Arguments:

        - time_series (numpy.ndarray): time series of number of infected
                                       persons
        - ax (axis): matplotlib axis to draw time series in
        """
        ax.plot(time_series, color='black')
        ax.set_xlabel('time')
        ax.set_ylabel('# infected')

    
    def plot_fatalities_time(self, time_series, ax):
        """
        Plots time number of fatalities over simulation time as a simple
        black line.

        Arguments:

        - time_series (numpy.ndarray): time series of number of 
                                       fatalities
        - ax (axis): matplotlib axis to draw time series in
        """
        ax.plot(time_series, color='black')
        ax.set_xlabel('time')
        ax.set_ylabel('# fatalities')
        if len(time_series) > 0:
            ax.text(0.05, 0.85, str(time_series[-1]), transform=ax.transAxes)


    def plot_immune_time(self, time_series, ax):
        """
        Plots time number of fatalities over simulation time as a simple
        black line.

        Arguments:

        - time_series (numpy.ndarray): time series of number of 
                                       fatalities
        - ax (axis): matplotlib axis to draw time series in
        """
        ax.plot(time_series, color='black')
        ax.set_xlabel('time')
        ax.set_ylabel('# immune')


class SimpleHealthSystemCurvesPlotter(DefaultCurvesPlotter):
    def __init__(self, health_system):
        """
        Curve plotter which draws additionally draws the threshold of a
        SimpleHealthSystem object.

        Arguments:

        - health_system (SimpleHealthSystem): the health system used in the
                                              simulation the results of which
                                              are being plotted
        """
        self._health_system = health_system


    def plot_infected_time(self, time_series, ax):
        """
        Plots time number of infected people over simulation time as a simple
        black line. Additionally, plots a red dashed line at the health system
        threshold.

        Arguments:

        - time_series (numpy.ndarray): time series of number of infected
                                       persons
        - ax (axis): matplotlib axis to draw time series in
        """
        ax.axhline(self._health_system.threshold, ls='--',
                   color='red')
        super().plot_infected_time(time_series, ax)


class Visualization(metaclass=ABCMeta):
    def __init__(self, simulation_results):
        """
        Abstract class defining the interface for simulation results
        visualizations.

        Arguments:
        
        - simulation_results (dict): simulation results as output by the run
                                     method of a simulation object
        """
        self._simulation_results = simulation_results
        self._check_result_shapes()


    def _check_result_shapes(self):
        """
        Checks whether shapes of simulation result arrays match
        """
        positions = self._simulation_results['all_positions']
        infected = self._simulation_results['all_infected']
        healthy = self._simulation_results['all_healthy']
        immune = self._simulation_results['all_immune']
        fatalities = self._simulation_results['all_fatalities']
        if not len(positions) == len(infected) == len(healthy) \
           == len(fatalities) == len(immune):
            raise ValueError("Simulation result arrays have unequal lengths")


    @property
    def figure(self):
        """
        The matplotlib figure object in which simulation results are drawn
        """
        return self._figure
        
        
    @abstractmethod
    def _setup_figure(self):
        """
        Sets up the matplotlib figure object.
        """
        pass
    
        
    @abstractmethod
    def visualize_single_step(self, step):
        """
        Visualizes a single step.

        Arguments:

        - step (int): simulation time step which to visualize
        """
        pass
        

class DefaultVisualization(Visualization):
    def __init__(self, simulation_results, geometry_drawer, persons_drawer,
                 curve_plotter, fig_args={}):
        """
        This default visualization draws the geometry and persons in a large
        subplot on the left part of the figure and the time series of infected,
        immune and deaths as a column of subplots on the right.

        Arguments:
        
        - simulation_results (dict): simulation results as output by the run
                                     method of a simulation object
        - geometry_drawer (GeometryDrawer): drawer object which draws the 
                                            geometry
        - persons_drawer (PersonsDrawer): drawer object which draws the persons
        - curve_plotter (CurvePlotter): plotter object which plots the time
                                        series of simulation macrostates
        - figure_args (dict): dictionary of arguments to be passed to the
                              plt.figure() call
        """
        super().__init__(simulation_results)
        self._geometry_drawer = geometry_drawer
        self._persons_drawer = persons_drawer
        self._curve_plotter = curve_plotter
        
        positions = self._simulation_results['all_positions']
        self._n_steps = positions.shape[0]
        self._n_persons = positions.shape[1]

        self._setup_figure(fig_args)
        self._setup_axes()


    def _setup_figure(self, fig_args):
        """
        Sets up the matplotlib figure object.

        Arguments:

        - fig_args (dict): keyword arguments to be passed on to the
                           plt.figure() call
        """
        self._figure = plt.figure(**fig_args)
    

    def _setup_axes(self):
        """
        Sets up the subplots
        """
        gs = gridspec.GridSpec(3, 4)

        main_ax = self._figure.add_subplot(gs[:,:3])
        self._geometry_drawer.draw(main_ax)
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


    def visualize_single_step(self, step):
        """
        Visualizes a single step.

        Arguments:

        - step (int): simulation time step which to visualize
        """
        positions = self._simulation_results['all_positions']
        infected = self._simulation_results['all_infected']
        immune = self._simulation_results['all_immune']
        fatalities = self._simulation_results['all_fatalities']
        healthy = self._simulation_results['all_healthy']

        currently_infected = positions[step, infected[step]]
        currently_healthy = positions[step, healthy[step]]
        currently_dead = positions[step, fatalities[step]]

        current_simulation_state = dict(
            infected=currently_infected,
            healthy=currently_healthy,
            dead=currently_dead
            )

        self._persons_drawer.draw(current_simulation_state, self._main_ax)
        self._curve_plotter.plot_infected_time(infected[:step].sum(1),
                                               self._infected_ax)
        self._curve_plotter.plot_immune_time(immune[:step].sum(1),
                                             self._immune_ax)
        self._curve_plotter.plot_fatalities_time(fatalities[:step].sum(1),
                                                 self._fatalities_ax)

        self.figure.tight_layout()
