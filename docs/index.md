# text2graph

A unified pipeline for text classification through graphs, for when there are no
labeled examples: an LLM labels a few documents, and graph models spread those
labels to the rest.

A `Pipeline` runs a sequence of components over a shared context dict. A typical
pipeline turns documents into nodes, embeds them, connects similar ones, picks a
few to label with one or several LLMs, and propagates those labels to the rest of
the graph with a GNN. Start with the [Quickstart](quickstart.md), and see every component in the
[API reference](reference/pipeline.md).

--8<-- "readme.md:install"

## Paper

text2graph started as the code for the paper
[Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification
in Label-Scarce Scenarios](https://ieeexplore.ieee.org/abstract/document/11264654).
If you use it, please cite:

```bibtex
@INPROCEEDINGS{11264654,
  author={Sarcinelli, João Lucas Luz Lima and Marcacini, Ricardo Marcondes},
  booktitle={2025 IEEE/SBC 37th International Symposium on Computer Architecture and High Performance Computing Workshops (SBAC-PADW)},
  title={Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios},
  year={2025},
  pages={124-130},
  doi={10.1109/SBAC-PADW69789.2025.00025}}
```
