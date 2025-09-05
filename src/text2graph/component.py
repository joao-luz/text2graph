class Component:
    name = ''
    def __init__(self, name):
        self.name = name
        self.str_parameters = {}

    def __str__(self):
        params_str = ', '.join(f'{key}={value}' for key,value in self.str_parameters.items())
        return self.name.replace('_', ' ').title().replace(' ', '') + f'({params_str})'