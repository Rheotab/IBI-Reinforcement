class Tracker():
    def __init__(self):
        self.qvalues = []
        self.actions = []

    def add_act(self, action):
        self.actions.append(action)

    def add_qvalues(self, qval):
        self.qvalues.append(qval)
