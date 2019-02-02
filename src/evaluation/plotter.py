from visdom import Visdom


class Plotter:

    def __init__(self, experiment_env):
        self.viz = Visdom()
        self.experiment_env = experiment_env
        self.plots = {}


    def plot(self, metric, data_type, title, epoch, value):

        if metric not in self.plots:
            pass
            # self.plots[metric] = self.viz.line(X=)