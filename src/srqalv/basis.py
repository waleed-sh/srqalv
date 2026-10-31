from itertools import combinations, product
import random
from types import SimpleNamespace

import numpy as np

from .recoupling import all_left_comb_channels, left_comb_valid_py


class K5Graph:

    def __init__(self, *, random_embedding=True, random_embedding_seed=1):
        if not random_embedding:
            raise ValueError("These experiments use a random embedding")
        rng = random.Random(random_embedding_seed)
        self.coordinates = np.asarray(
            [[float(f"{rng.gauss(0.0, 5.0):.12g}") for _ in range(3)] for _ in range(5)]
        )
        self.edges = tuple(combinations(range(5), 2))


def orientation_sign(graph, vertex, edges):
    tangents = []
    for first, second in edges:
        if vertex not in (first, second):
            raise ValueError("Edge is not incident on the vertex")
        other = second if first == vertex else first
        tangents.append(graph.coordinates[other] - graph.coordinates[vertex])
    return int(np.sign(np.linalg.det(np.asarray(tangents))))


class SU2Hilbert:

    def __init__(self, graph, *, two_j_max, exact=True, max_exact_states=100_000, random_init=None):
        self.two_j_max = int(two_j_max)
        if self.two_j_max < 0:
            raise ValueError("two_j_max must be nonnegative")
        self.n_edges = 10
        self.n_vertices = 5
        self.max_exact_states = int(max_exact_states)
        incidence = tuple(
            tuple(e for e, edge in enumerate(graph.edges) if vertex in edge)
            for vertex in range(5)
        )
        self.metadata = SimpleNamespace(incident_edges_by_vertex=incidence, edge_tokens=graph.edges)
        self.layout = SimpleNamespace(
            vertex_ids=tuple(range(5)),
            vertex_channel_sites=tuple((10 + v,) for v in range(5)),
            vertex_channel_max_labels=((2 * self.two_j_max,),) * 5,
        )

    def all_states(self, *, max_states=None):
        if self.two_j_max > 2:
            raise ValueError("Full K5 enumeration is restricted to the two validation cutoffs")
        limit = self.max_exact_states if max_states is None else int(max_states)
        states = []
        # keep edge sectors and channel tuples in the original lexicographic order
        for edge_values in product(range(self.two_j_max + 1), repeat=10):
            choices = []
            for vertex, edges in enumerate(self.metadata.incident_edges_by_vertex):
                channels = all_left_comb_channels(
                    tuple(edge_values[e] for e in edges), self.layout.vertex_channel_max_labels[vertex]
                )
                if not channels:
                    break
                choices.append(channels)
            else:
                for channels in product(*choices):
                    states.append((*edge_values, *(c[0] for c in channels)))
                    if len(states) > limit:
                        raise ValueError("Exact enumeration exceeded max_states")
        return np.asarray(states, dtype=np.int32).reshape(-1, 15)
