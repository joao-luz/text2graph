import logging

from .component_registry import register_component


logger = logging.getLogger(__name__)


class Component:
    """Base class for pipeline steps.

    A component is called with the pipeline's shared `context` dict and returns it,
    usually after adding to it or changing the graph in it. Keys shared by the
    built-in components:

    - `graph`: the `torch_geometric` `Data` (one node type) or `HeteroData` graph
      being built. Created by the first node-creation step, which turns it into
      `HeteroData` once a second node type is added.
    - `node_types`: names of the node types in the graph, in the order they were added.
    - `label_info`: one dict per node of the labeled node type, describing how its
      label was obtained, e.g. `{'source': 'gcn_propagator', 'prob': 0.93}`
      (empty for unlabeled nodes).

    Components also read keys named by their parameters, e.g. a sampler's
    `mask_name` or `DocumentsToNodes`' `source`.

    To write one, subclass this, implement `run`, and set `self.str_parameters` in
    the constructor: it's what `str(component)` shows, and the pipeline logs it.
    Register the class with `register_component` to use it in pipeline configs::

        @register_component('drop_short_documents')
        class DropShortDocuments(Component):
            def __init__(self, min_words=5):
                self.min_words = min_words
                self.str_parameters = {'min_words': min_words}

            def run(self, context):
                context['documents'] = [d for d in context['documents'] if len(d.split()) >= self.min_words]
                return context
    """

    def run(self, context, **kwargs):
        """Run the step.

        Args:
            context: The pipeline's shared context dict.

        Returns:
            The context, for the next step. Return the same dict after changing it.
        """
        return context

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __str__(self):
        params_str = ', '.join(f'{key}={value}' for key,value in self.str_parameters.items())
        return type(self).__name__ + f'({params_str})'


@register_component('repeat_component')
class RepeatComponent(Component):
    """Runs a list of steps `n` times in a row.

    In a config, the nested `steps` are written like the pipeline's own::

        {'name': 'repeat_component',
         'parameters': {
             'n': 3,
             'steps': [
                 {'name': 'kmeans_sampler', 'parameters': {'n': 50}},
                 {'name': 'llm_labeler', 'parameters': {'model': '${llms[0]}', ...}},
             ],
         }}

    Args:
        steps: List of components to run, in order, on each repetition.
        n: Number of repetitions.
        verbose: Log each step at INFO level, with the repetition number. A
            `Pipeline` overrides this with its own `verbose` when it runs.
    """

    def __init__(self, steps, n, verbose=True):
        self.steps = steps
        self.n = n
        self.verbose = verbose

    def run(self, context):
        for i in range(self.n):
            for step in self.steps:
                if self.verbose:
                    logger.info('%s (%d)...', step, i)
                context = step(context)

        return context

    def __str__(self):
        steps_str = '\n'.join(['\t' + line for line in ',\n'.join([str(step) for step in self.steps]).split('\n')])
        return type(self).__name__ + f'(n={self.n}, steps=[\n' + steps_str + '\n])'
