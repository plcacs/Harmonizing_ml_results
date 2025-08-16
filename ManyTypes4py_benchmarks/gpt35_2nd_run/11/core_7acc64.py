import typing as tp
import networkx as nx
import cdt
from cdt.metrics import precision_recall, SHD
import nevergrad as ng
from ..base import ExperimentFunction

class CausalDiscovery(ExperimentFunction):
    def __init__(self, generator: str = 'sachs', causal_mechanism: str = 'linear', noise: tp.Union[str, tp.Callable] = 'gaussian', noise_coeff: float = 0.4, npoints: int = 500, nodes: int = 20, parents_max: int = 5, expected_degree: int = 3) -> None:
        _dataset: tp.List[str] = ['tuebingen', 'sachs', 'dream4']
        assert generator in _dataset + ['acylicgraph']
        assert causal_mechanism in ['linear', 'polynomial', 'sigmoid_add', 'sigmoid_mix', 'gp_add', 'gp_mix', 'nn']
        if generator in _dataset:
            self._data, self._ground_truth_graph = cdt.data.load_dataset(generator)
        else:
            _generator = cdt.data.AcyclicGraphGenerator(causal_mechanism=causal_mechanism, noise=noise, noise_coeff=noise_coeff, npoints=npoints, nodes=nodes, parents_max=parents_max, expected_degree=expected_degree)
            self._data, self._ground_truth_graph = _generator.generate()
        self._nodes_list: tp.List[str] = list(self._ground_truth_graph.nodes())
        self._nvars: int = self._data.shape[1]
        param_links = ng.p.Choice([-1, 0, 1], repetitions=self._nvars * (self._nvars - 1) // 2)
        instru = ng.p.Instrumentation(network_links=param_links).set_name('')
        super().__init__(self.objective, instru)

    def objective(self, network_links: tp.List[int]) -> float:
        output_graph = self.choices_to_graph(network_links)
        score = self.graph_score(output_graph)
        return -score

    def graph_score(self, test_graph: nx.DiGraph) -> float:
        pr_score, _ = precision_recall(self._ground_truth_graph, test_graph)
        shd_score = SHD(self._ground_truth_graph, test_graph)
        return float(pr_score - shd_score)

    def add_missing_nodes(self, graph: nx.DiGraph) -> nx.DiGraph:
        for n in set(self._nodes_list) - set(graph.nodes()):
            graph.add_node(n)
        return graph

    def choices_to_graph(self, network_links: tp.List[int]) -> nx.DiGraph:
        output_graph = nx.DiGraph()
        k = 0
        for i in range(1, self._nvars):
            for j in range(i + 1, self._nvars):
                if network_links[k] == -1:
                    output_graph.add_edge(self._nodes_list[j], self._nodes_list[i])
                elif network_links[k] == +1:
                    output_graph.add_edge(self._nodes_list[i], self._nodes_list[j])
                k += 1
        return self.add_missing_nodes(output_graph)
