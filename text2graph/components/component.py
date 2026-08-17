from .component_registry import register_component


class Component:
    def run(self, context, **kwargs):
        return context

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __str__(self):
        params_str = ', '.join(f'{key}={value}' for key,value in self.str_parameters.items())
        return type(self).__name__ + f'({params_str})'


@register_component('repeat_component')
class RepeatComponent(Component):
    def __init__(self, steps, n, verbose=True):
        self.steps = steps
        self.n = n
        self.verbose = verbose

    def run(self, context):
        for i in range(self.n):
            for step in self.steps:
                if self.verbose:
                    print(str(step) + f' ({i})...')
                context = step(context)

        return context

    def __str__(self):
        steps_str = '\n'.join(['\t' + line for line in ',\n'.join([str(step) for step in self.steps]).split('\n')])
        return type(self).__name__ + f'(n={self.n}, steps=[\n' + steps_str + '\n])'