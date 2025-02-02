import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def interp_x_u_history(ts, xs, us, ts_query):
    right = np.minimum(len(ts) - 1, np.searchsorted(ts, ts_query, side='right'))  # First index where times[i] > t_query
    left = np.maximum(0, right - 1)  # Previous index
    
    alphas = (ts_query - ts[left]) / (ts[right] - ts[left])
    
    x_query = xs[left] + alphas[:, np.newaxis] * (xs[right] - xs[left])
    u_query = us[left] + alphas[:, np.newaxis] * (us[right] - us[left])
    
    return ts_query, x_query, u_query


class RenderEnvironment:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.elements = []
    
    def add_element(self, element, *args, **kwargs):
        self.elements.append(element(self, *args, **kwargs))
        

    def render(self, t_range, t_history, X_history, U_history,
               fps=30, repeat=True):
        import matplotlib.animation as animation

        t_render = np.arange(t_range[0], t_range[1], 1/fps)
        _, x_render, u_render = interp_x_u_history(t_history, X_history, U_history, t_render)
        
        def update(idx):
            self.fig.suptitle(r"$t = {:.2f}$ s".format(t_render[idx]))
            
            for element in self.elements:
                element.update(
                    t_render[idx], x_render[idx], u_render[idx]
                )
        
        ani = animation.FuncAnimation(self.fig, update,
                                      frames=len(t_render),
                                      interval=1/fps * 1e3,
                                      repeat=repeat)
        plt.show()

class RenderElement:
    def __init__(self, env: RenderEnvironment) -> None:
        self.env = env
    
    def update(self, t, x, u): raise NotImplementedError