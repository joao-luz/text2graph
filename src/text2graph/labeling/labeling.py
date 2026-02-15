from ..component import Component


class Labeler(Component):
    def __init__(self, name, sample_mask_attribute='sample_mask'):
        super().__init__(name)

        self.sample_mask_attribute = sample_mask_attribute

    def label(self, inputs):
        pass

    def __call__(self):
        pass 
