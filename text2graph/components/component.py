from .component_factory import register_component, component_from_config


class Component:
    name = ''
    def __init__(self, name):
        self.name = name
        self.str_parameters = {}

    def __str__(self):
        params_str = ', '.join(f'{key}={value}' for key,value in self.str_parameters.items())
        return self.name.replace('_', ' ').title().replace(' ', '') + f'({params_str})'
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        pass
    
    @classmethod
    def from_config(cls, params, **kwargs):
        return cls(**params)
    

class RepeatComponent(Component):
    def __init__(self, steps, repeat):
        super().__init__('repeat_component')

        self.steps = steps
        self.repeat = repeat

        self.str_parameters = {
            'repeat': repeat,
            'steps': steps
        }

    def forward(self, data, *args, **kwargs):
        data = data.clone()
        for _ in range(self.repeat):
            for step in self.steps:
                data = step(data, *args, **kwargs)

        return data
    
    @classmethod
    def from_config(cls, params, **kwargs):
        steps = []
        for step_config in params['steps']:
            step = component_from_config(step_config, **kwargs)
            steps.append(step)
        params['steps'] = steps

        return cls(**params)
