import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def interp_x_u_history(ts, xs, us, ts_query):
    right = np.minimum(len(ts) - 1, np.searchsorted(ts, ts_query, side='right'))  # First index where times[i] > t_query
    left = np.maximum(0, right - 1)  # Previous index
    
    alphas = (ts_query - ts[left]) / (ts[right] - ts[left])
    
    x_query = xs[left] + alphas[:, np.newaxis] * (xs[right] - xs[left])
    
    us_left = np.maximum(0, np.minimum(len(us)-1, left))
    us_right = np.maximum(0, np.minimum(len(us)-1, right))
    
    u_query = us[us_left] + alphas[:, np.newaxis] * (us[us_right] - us[us_left])
    
    return ts_query, x_query, u_query


class RenderEnvironment:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.elements = []
    
    def add_element(self, element, *args, **kwargs):
        self.elements.append(element(self, *args, **kwargs))
        
    def render(self, t_range,
               fps=30, repeat=True, save_fpath=None):
        import matplotlib.animation as animation

        n_frames = int(fps * (t_range[1] - t_range[0]))
        actual_fps = int(n_frames / (t_range[1] - t_range[0]))
        
        t_render = np.linspace(t_range[0], t_range[1], n_frames)
        
        def update(idx):
            self.fig.suptitle(r"$t = {:.2f}$ s".format(t_render[idx]))
            
            for element in self.elements:
                element.update(t_render[idx])
        
        ani = animation.FuncAnimation(self.fig, update,
                                      frames=n_frames,
                                      interval=1/actual_fps * 1e3,
                                      repeat=repeat)

        if save_fpath is not None:
            base, _ = os.path.splitext(save_fpath)
            fpath = f"{base}.mp4"
            
            print(f"Saving animation to {fpath} ...")
            ani.save(fpath, writer=animation.FFMpegWriter(fps=actual_fps, codec="mpeg4"))
        
        return ani
        
class RenderElement:
    def __init__(self, env: RenderEnvironment) -> None:
        self.env = env
    
    def update(self, t, x, u): raise NotImplementedError