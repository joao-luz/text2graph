from .component import Component

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import pickle
import copy

class GraphVisualizer(Component):
    def __init__(self, name, output_file=None):
        super().__init__(name)

        self.output_dir = '.'
        self.output_file = output_file

        self.str_parameters = {
            'output_file': f'{self.output_dir}/{self.output_file}'
        }

    def set_output_dir(self, dir):
        self.output_dir = dir

    def __call__(self, G):
        pass
    
class FigureVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.png'
        super().__init__('figure_visualizer', output_file)

    def visualize(self, G):
        classes = list(set([G.nodes[n].get('label') for n in G.nodes if G.nodes[n].get('label') is not None]))

        if len(classes) <= 10:
            colors = mpl.colormaps['tab10'].colors
        elif len(classes) <= 20:
            colors = mpl.colormaps['tab20'].colors
        else:
            colors = mpl.cm.get_cmap('rainbow')(np.linspace(0, 1, len(classes)))

        label2color = {name: colors[classes.index(name)] for name in classes}

        fig,ax = plt.subplots(figsize=(8, 8))

        handles = []
        for label,color in label2color.items():
            handles.append(mpatches.Patch(color=color, label=label))
        fig.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left")
        fig.tight_layout()

        node_colors = []
        for n in G.nodes:
            label = G.nodes[n].get('label')
            color = 'lightgrey' if label is None else label2color[label]
            node_colors.append(color)

        node_sizes = [20 if G.nodes[n].get('label') is not None else 10 for n in G.nodes]
        nx.draw_networkx(G, with_labels=False, font_size=9, node_color=node_colors, ax=ax, node_size=node_sizes, width=0.1)

        return fig
    
    def __call__(self, G):
        fig = self.visualize(G)

        if self.output_file:
            fig.savefig(f'{self.output_dir}/{self.output_file}.png', bbox_inches='tight')

        return G

def _remove_attributes(G, attributes):
    if not isinstance(attributes, list):
        attributes = [attributes]

    for _, data in G.nodes(data=True):
        for attr in attributes:
            if attr in data:
                del data[attr]

    return G

class GMLVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.gml'
        super().__init__('gml_visualizer', output_file)

    def __call__(self, G):
        G_copy = _remove_attributes(copy.deepcopy(G), 'embedding')

        nx.write_gml(G_copy, f'{self.output_dir}/{self.output_file}')

        return G
    
class PickleVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.pkl'
        super().__init__('pickle_visualizer', output_file)

    def __call__(self, G):
        G_copy = _remove_attributes(copy.deepcopy(G), 'embedding')

        with open(f'{self.output_dir}/{self.output_file}', 'wb') as f:
            pickle.dump(G_copy, f)

        return G