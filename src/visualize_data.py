import matplotlib as plt
from matplotlib import animation


class RtDrawer():
    def __init__(self):
        fig, self.ax = plt.subplots()
        self.t = []
        self.x = []
        self.ln, = plt.plot([], [])

    def init_ani(self, xlim=[0, 256], ylim=[-3, 3]):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        return self.ln, 

    def update_ani(self, ti, xi):
        self.t.append(ti)
        self.x.append(xi)
        self.ln.set_data(ti, xi)
        return self.ln,

    def update(self, ti, xi, ):

