"""
Animator classes which implement different ways of animating simulation
visualization results.
"""
from abc import abstractmethod

from celluloid import Camera


class Animator:
    @abstractmethod
    def __init__(self, visualization):
        """
        Abstract class defining the interface for animators.
        
        Arguments:
        
        - visualization (Visualization): a Visualization object which plots
                                         simulation results for a single time
                                         step
        """
        self._visualization = visualization
        

    @abstractmethod
    def animate(self):
        """
        Loops over simulation steps and outputs a sequence of plots in some
        form.
        """
        pass


class CelluloidAnimator(Animator):
    def __init__(self, visualization, frame_rate=10, out="output.mp4"):
        """
        Animates simulation results using the celluloid Python package and
        outputs results as a MP4 file.

        Arguments:

        - visualization (Visualization): a Visualization object which plots
                                         simulation results for a single time
                                         step
        - frame_rate (int): the desired frame rate for the MP4 video
        - out (str): output filename
        """
        super(CelluloidAnimator, self).__init__(visualization)
        self._frame_rate = frame_rate
        self._out = out
        
        
    def animate(self, n_steps, start=0, interval=1):
        """
        Loops over simulation steps and creates and writes a MP4 video.
        
        Arguments:
        
        - n_steps (int): last simulation step to animate
        - start (int): first simulation step to animate
        - interval (int): allows to skip simulation steps. Only every
                          interval-nth step is animated. Default is 1,
                          meaning that all steps are animated.
        """
        camera = Camera(self._visualization.figure)
        for step in range(start, n_steps, interval):
            if step % 50 == 0:
                print(f"Animating step {step}/{n_steps}...")
            self._visualization.visualize_single_step(step)
            camera.snap()
        anim = camera.animate(blit=True)
        anim.save(self._out, fps=self._frame_rate)

        print("Done.")
