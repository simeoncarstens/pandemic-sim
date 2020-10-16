from abc import abstractmethod

from celluloid import Camera


class Animator:
    @abstractmethod
    def __init__(self, visualization):
        self._visualization = visualization
        

    @abstractmethod
    def animate(self):
        pass


class CelluloidAnimator(Animator):
    def __init__(self, visualization, frame_rate=10, out="output.mp4"):
        super(CelluloidAnimator, self).__init__(visualization)
        self._frame_rate = frame_rate
        self._out = out
        
        
    def animate(self, n_steps, start=0, interval=1):
        camera = Camera(self._visualization.figure)
        for step in range(start, n_steps, interval):
            if step % 50 == 0:
                print(f"Animating step {step}/{n_steps}...")
            self._visualization.visualize_single_step(step)
            camera.snap()
        anim = camera.animate(blit=True)
        anim.save(self._out, fps=self._frame_rate)

        print("Done.")
