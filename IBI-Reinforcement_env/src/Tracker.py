import matplotlib.pyplot as plt


class Tracker():
    def __init__(self, show, show_interval=None):
        self.qvalues = []
        self.actions = []
        self.losses = []
        self.scores = []
        self.show = show
        self.show_interval = show_interval
        if show:
            self.fig, self.axes = plt.subplots(2, 2)
            self.axes[0, 0].set_title("Actions")
            self.axes[0, 1].set_title("Qval")
            self.axes[1, 0].set_title("Score")
            self.axes[1, 1].set_title("Loss")
            self.colormap = ['r', 'g', 'b', 'y']
        # self.ax_act = axes[0, 0], self.ax_qval = axes[0, 1], self.ax_score = axes[1, 0], self.ax_loss = axes[1, 1]

    def add_act(self, action):
        self.actions.append(action)
        if self.show:
            x = [i for i in range(len(self.actions))]
            color = [self.colormap[self.actions[i]] for i in range(len(self.actions))]
            self.axes[0, 0].scatter(x, self.actions, color=color)
            plt.draw()
            plt.pause(0.00000001)

    def add_qvalues(self, qval):
        q = qval.tolist()
        q = q[0]
        self.qvalues.append(q)
        if self.show:
            x = [i for i in range(len(self.qvalues))]
            nb_qval = len(self.qvalues[0])
            for j in range(nb_qval):
                y = [self.qvalues[i][j] for i in range(len(self.qvalues))]
                # color = [self.colormap[i] for i in range(len(self.qvalues))]
                self.axes[0, 1].scatter(x, y, color=self.colormap[j])
            plt.draw()
            plt.pause(0.00000001)

    def add_score(self, score):
        self.scores.append(score)
        if self.show:
            x = [i for i in range(len(self.scores))]
            self.axes[1, 0].plot(x, self.scores)
            plt.draw()
            plt.pause(0.00000001)

    def add_losses(self, loss):
        loss_val = loss.item()
        self.losses.append(loss_val)
        if self.show:
            x = [i for i in range(len(self.losses))]
            self.axes[1, 1].plot(x, self.losses)
            plt.draw()
            plt.pause(0.00000001)
