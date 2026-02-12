from .component import Component

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import pickle
from torch_geometric.utils import to_networkx

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

    def __call__(self, data):
        pass
    
class FigureVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.png'
        super().__init__('figure_visualizer', output_file)

    def visualize(self, G):
        id2label = G.graph['id2label']
        classes = list(set([id2label[G.nodes[n].get('y')] for n in G.nodes if G.nodes[n].get('y') != -1]))

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
            label = id2label[G.nodes[n].get('y')]
            color = 'lightgrey' if label is None else label2color[label]
            node_colors.append(color)

        node_sizes = [20 if G.nodes[n].get('y') != -1 else 10 for n in G.nodes]
        nx.draw_networkx(G, with_labels=False, font_size=9, node_color=node_colors, ax=ax, node_size=node_sizes, width=0.1)

        return fig
    
    def __call__(self, data):
        G = to_networkx(data, node_attrs=['text', 'y', 'label_info'], edge_attrs=['edge_weight'], graph_attrs=['id2label'])

        fig = self.visualize(G)

        if self.output_file:
            fig.savefig(f'{self.output_dir}/{self.output_file}.png', bbox_inches='tight')

        return data

class GMLVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.gml'
        super().__init__('gml_visualizer', output_file)

    def __call__(self, data):
        G = to_networkx(data, node_attrs=['text', 'y', 'label_info'], edge_attrs=['edge_weight'])

        nx.write_gml(G, f'{self.output_dir}/{self.output_file}')

        return data
    
class PickleVisualizer(GraphVisualizer):
    def __init__(self, output_file=None):
        output_file = output_file + '.pkl'
        super().__init__('pickle_visualizer', output_file)

    def __call__(self, data):
        G = to_networkx(data, node_attrs=['text', 'y', 'label_info'], edge_attrs=['edge_weight'])

        with open(f'{self.output_dir}/{self.output_file}', 'wb') as f:
            pickle.dump(G, f)

        return data