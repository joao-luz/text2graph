from ..component import Component


class LabelPropagator(Component):
    def __init__(self, name):
        super().__init__(name)

    def propagate(self, data):
        pass