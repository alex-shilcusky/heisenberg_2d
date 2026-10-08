from collections.abc import Sequence

import numpy as np

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert
from netket.utils.types import DType

from netket.operator._graph_operator import GraphOperator
import time
import jax 
import netket as nk
import jax.numpy as jnp


import igraph

from netket.utils.deprecation import warn_deprecation
from netket.utils.group import Permutation, PermutationGroup
from netket.graph.abstract_graph import AbstractGraph, Edge, ColoredEdge, EdgeSequence

from jax.tree_util import register_pytree_node_class

# just changed igraph.Graph(directed=False) to igraph.Graph(directed=True)
class GraphCust(AbstractGraph):
    """
    A simple implementation of Graph based on an external graph library.

    The underlying implementation is based on igraph and supports conversion to
    networkx, but this is an implementation detail and could be changed in the future.
    """

    # Initialization
    # ------------------------------------------------------------------------
    def __init__(
        self,
        edges: Sequence[Edge] | Sequence[ColoredEdge],
        n_nodes: int | None = None,
    ):
        """
        Construct the a graph starting from a list of edges and optionally a given
        number of nodes.

        Args:
            edges: list of (undirected) edges
            n_nodes: number of nodes. Can be used to specify the number vertices in the
                graph if not all vertices appear in an edge.

        """
        edges, colors = self._clean_edges(edges)
        # print('edges = ', edges)
        # print('colors = ', colors)

        if n_nodes is None:
            if len(edges) > 0:
                n_nodes = max(max(e) for e in edges) + 1
            else:
                n_nodes = 0

        graph = igraph.Graph(directed=True)
        graph.add_vertices(n_nodes)
        graph.add_edges(edges, attributes={"color": colors})

        self._igraph = graph
        self._automorphisms = None

    @staticmethod
    def _clean_edges(edges):
        """Validate and normalize edges argument."""
        if not isinstance(edges, Sequence):
            raise TypeError("edges must be a sequence.")
        if len(edges) == 0:
            return edges, []

        e0 = edges[0]
        if not isinstance(e0, Sequence) or len(e0) not in (2, 3):
            raise ValueError(
                "Edges must be tuple of length 2 (or 3 for colored edges)."
            )
        if not all(len(e) == len(e0) for e in edges):
            raise ValueError("Either all or none of the edges need to specify a color.")

        if len(e0) == 2:
            return edges, [0] * len(edges)
        else:
            return [(v, w) for (v, w, _) in edges], [c for (*_, c) in edges]

    # Conversion
    # ------------------------------------------------------------------------
    @classmethod
    def from_igraph(cls, graph: igraph.Graph) -> "Graph":
        """
        Creates a new Graph instance from an igraph.Graph instance.
        """
        if graph.is_directed():
            raise ValueError("Directed graphs are not currently supported.")
        self = cls.__new__(cls)
        self._igraph = graph.copy()
        self._automorphisms = None

        if "color" not in self._igraph.edge_attributes():
            self._igraph.es.set_attribute_values("color", [0] * self._igraph.ecount())
        else:
            if not all(isinstance(c, int) for c in self.edge_colors):
                raise ValueError(
                    "graph has 'color' edge attributes, but not all colors are integers."
                )

        return self

    @classmethod
    def from_networkx(cls, graph) -> "Graph":
        """
        Creates a new Graph instance from a networkx graph.
        """
        ig = igraph.Graph.from_networkx(graph)
        return cls.from_igraph(ig)

    def to_igraph(self):
        """
        Returns a copy of this graph as an igraph.Graph instance.
        """
        return self._igraph.copy()

    def to_networkx(self):
        """
        Returns a copy of this graph as an igraph.Graph instance.
        This method requires networkx to be installed.
        """
        return self._igraph.to_networkx()

    # Graph properties
    # ------------------------------------------------------------------------
    def adjacency_list(self) -> list[list]:
        return self._igraph.get_adjlist()

    def is_connected(self) -> bool:
        return self._igraph.is_connected()

    def is_bipartite(self) -> bool:
        return self._igraph.is_bipartite()

    @property
    def n_nodes(self) -> int:
        r"""The number of nodes (or vertices) in the graph"""
        return self._igraph.vcount()

    @property
    def n_edges(self):
        r"""The number of edges in the graph."""
        return self._igraph.ecount()

    def nodes(self) -> Sequence[int]:
        return range(self._igraph.vcount())

    def edges(
        self,
        color=None,
        *,
        return_color: bool = False,
        filter_color: int | None = None,
    ) -> EdgeSequence:
        if color is not None:
            warn_deprecation(
                "The color option has been split into return_color and filter_color."
            )
            # need to check for bool first, because bool is a subclass of int
            if isinstance(color, bool):
                return_color = color
            elif isinstance(color, int):
                filter_color = color
            else:
                raise TypeError("Incorrect type for 'color'")

        if not return_color and filter_color is None:
            return self._igraph.get_edgelist()

        edges_with_color = zip(self._igraph.get_edgelist(), self.edge_colors)
        if filter_color is not None:
            edges_with_color = filter(lambda x: x[1] == filter_color, edges_with_color)
        if return_color:
            return [(*e, c) for (e, c) in edges_with_color]
        else:
            return [e for (e, _) in edges_with_color]

    @property
    def edge_colors(self) -> Sequence[int]:
        r"""Sequence of edge colors, in the order of the edges returned by
        :code:`self.edges`."""
        if self.n_edges > 0:
            return self._igraph.es.get_attribute_values("color")
        else:
            return []

    # Graph algorithms
    # ------------------------------------------------------------------------
    def distances(self) -> list[list]:
        return np.array(self._igraph.distances())

    def _compute_automorphisms(self):
        """
        Compute the graph autmorphisms of this graph.
        """
        colors = self.edge_colors
        result = self._igraph.get_isomorphisms_vf2(
            edge_color1=colors, edge_color2=colors
        )

        # sort them s.t. the identity comes first
        result = np.unique(result, axis=0).tolist()
        result = PermutationGroup([Permutation(i) for i in result], self.n_nodes)
        return result

    # TODO turn into a struct.property_cached?
    def automorphisms(self) -> PermutationGroup:
        if self._automorphisms is None:
            self._automorphisms = self._compute_automorphisms()
        return self._automorphisms

    # Output and drawing
    # ------------------------------------------------------------------------
    def __repr__(self):
        return "{}(n_nodes={}, n_edges={})".format(
            str(type(self)).split(".")[-1][:-2], self.n_nodes, self.n_edges
        )


def Edgeless(n_nodes: int) -> Graph:
    """
    Construct a set graph (collection of unconnected vertices).

    Args:
        nodes: An integer number of nodes or a list of ints that index nodes of a graph.

    Example:
        >>> import netket
        >>> g=netket.graph.Edgeless([0,1,2,3])
        >>> print(g.n_nodes)
        4
        >>> print(g.n_edges)
        0
    """
    return Graph([], n_nodes)


def DoubledGraph(graph: AbstractGraph) -> Graph:
    """
    DoubledGraph(graph)

    Constructs a DoubledGraph representing the doubled hilbert space of a density
    operator. The resulting graph is composed of two disjoint sub-graphs identical
    to the input.

    Args:
        graph: The graph to double
    """

    dedges = list(graph.edges())
    n_v = graph.n_nodes

    dnodes = 2 * n_v
    dedges += [(edge[0] + n_v, edge[1] + n_v) for edge in graph.edges()]

    return Graph(n_nodes=dnodes, edges=dedges)


def disjoint_union(*graphs: tuple[Graph, ...]) -> Graph:
    """
    Returns the ordered disjoint union of the input graphs.

    The ordering of the nodes is the same as the input order, so if
    the I-th input graph has :math:`n_i` nodes, the first `[0, n_1[` nodes
    will correspond to the first graph, the nodes between :math:`[n_1, n_1+n2[`
    will correspond to the second graph, and so on and so forth.

    Args:
       *graphs: A variable number of NetKet graphs (or lattices).

    Output:
        A NetKet Graph object.
    """
    if len(graphs) == 0:
        raise ValueError("Must provide at least one graph")
    elif len(graphs) == 1:
        return graphs[0]
    else:
        return Graph.from_igraph(igraph.disjoint_union([g._igraph for g in graphs]))
    


############################################################################################################
from netket.operator._local_operator import LocalOperator

def check_acting_on_subspace(acting_on_subspace, hilbert, graph):
    """Check `acting_on_subspace` argument used by various operators."""
    if acting_on_subspace is None:
        acting_on_subspace = list(range(hilbert.size))
    elif isinstance(acting_on_subspace, int):
        start = acting_on_subspace
        acting_on_subspace = [start + i for i in range(graph.n_nodes)]
    elif isinstance(acting_on_subspace, list):
        if len(acting_on_subspace) != graph.n_nodes:
            raise ValueError(
                "acting_on_subspace must be a list of length graph.n_nodes"
            )
    else:
        raise TypeError("acting_on_subspace must be a list or single integer")

    return acting_on_subspace


# GraphOperatorCust is redundant
@register_pytree_node_class
class GraphOperatorCust(LocalOperator):
    r"""
    A graph-based quantum operator. In its simplest terms, this is the sum of
    local operators living on the edge of an arbitrary graph.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        site_ops=[],
        bond_ops=[],
        bond_ops_colors=[],
        dtype: DType | None = None,
        *,
        acting_on_subspace: None | list[int] | int = None,
    ):
        r"""
        A graph-based quantum operator. In its simplest terms, this is the sum of
        local operators living on the edge of an arbitrary graph.

        A ``GraphOperator`` is constructed giving a hilbert space and either a
        list of operators acting on sites or a list acting on the bonds.
        Users can specify the color of the bond that an operator acts on, if
        desired. If none are specified, the bond operators act on all edges.

        Args:
         hilbert: Hilbert space the operator acts on.
         graph: The graph whose vertices and edges are considered to construct the
                operator
         site_ops: A list of operators in matrix form that act
                on the nodes of the graph.
                The default is an empty list. Note that if no site_ops are
                specified, the user must give a list of bond operators.
         bond_ops: A list of operators that act on the edges of the graph.
             The default is None. Note that if no bond_ops are
             specified, the user must give a list of site operators.
         bond_ops_colors: A list of edge colors, specifying the color each
             bond operator acts on. The default is an empty list.
         dtype: Data type of the matrix elements.
         acting_on_subspace: Specifies the mapping between nodes of the graph and
            Hilbert space sites, so that graph node :code:`i ∈ [0, ..., graph.n_nodes]`,
            corresponds to :code:`acting_on_subspace[i] ∈ [0, ..., hilbert.n_sites]`.
            Must be a list of length `graph.n_nodes`. Passing a single integer :code:`start`
            is equivalent to :code:`[start, ..., start + graph.n_nodes - 1]`.

        Examples:
         Constructs a ``GraphOperator`` operator for a 2D system.

         >>> import netket as nk
         >>> sigmax = [[0, 1], [1, 0]]
         >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
         >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
         >>> g = nk.graph.Graph(edges=edges)
         >>> hi = nk.hilbert.Spin(0.5, N=g.n_nodes)
         >>> op = nk.operator.GraphOperator(
         ... hi, site_ops=[sigmax], bond_ops=[mszsz], graph=g)
         >>> print(op)
         GraphOperator(dim=20, #acting_on=40 locations, constant=0.0, dtype=float64, graph=Graph(n_nodes=20, n_edges=20))
        """
        acting_on_subspace = check_acting_on_subspace(
            acting_on_subspace, hilbert, graph
        )
        self._acting_on_subspace = acting_on_subspace

        # Ensure that at least one of SiteOps and BondOps was initialized
        if len(bond_ops) == 0 and len(site_ops) == 0:
            raise ValueError("Must input at least site_ops or bond_ops.")

        # Create the local operator as the sum of all site and bond operators
        operators = []
        acting_on = []

        # Site operators
        if len(site_ops) > 0:
            for i in range(graph.n_nodes):
                for site_op in site_ops:
                    i_prime = acting_on_subspace[i]
                    operators.append(site_op)
                    acting_on.append([i_prime])

        # Bond operators
        if len(bond_ops_colors) > 0:
            if len(bond_ops) != len(bond_ops_colors):
                raise ValueError(
                    """The GraphHamiltonian definition is inconsistent.
                    The sizes of bond_ops and bond_ops_colors do not match."""
                )

            if len(bond_ops) > 0:
                #  Use edge_colors to populate operators
                for u, v, color in graph.edges(return_color=True):
                    u, v = acting_on_subspace[u], acting_on_subspace[v]
                    for c, bond_color in enumerate(bond_ops_colors):
                        if bond_color == color:
                            operators.append(bond_ops[c])
                            acting_on.append([u, v])
        elif len(bond_ops) > 0:
            assert len(bond_ops) == 1

            for u, v in graph.edges():
                u, v = acting_on_subspace[u], acting_on_subspace[v]
                operators.append(bond_ops[0])
                acting_on.append([u, v])

        super().__init__(hilbert, operators, acting_on, dtype=dtype)
        self._graph = graph

    @property
    def graph(self) -> AbstractGraph:
        """The graph on which this Operator is defined"""
        return self._graph

    @property
    def acting_on_subspace(self):
        """
        Mapping between nodes of the graph and Hilbert space sites as given in
        the constructor.
        """
        return self._acting_on_subspace

    def copy(self, *, dtype: DType | None = None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype
        """
        return super().copy(dtype=dtype, _cls=LocalOperator)

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, constant={self.constant}, dtype={self.dtype}, graph={self.graph})"

    def tree_flatten(self):
        return super().tree_flatten()

    @classmethod
    def tree_unflatten(cls, metadata, data):
        return LocalOperator.tree_unflatten(metadata, data)

######################################################
def get_cluster_graph(L ,PBC):
    graph1 = nk.graph.Hypercube(length=L, n_dim=1, pbc=PBC) # graph of clusters (each site is ONE cluster!)
    edges = graph1.edges()
    cust_edges = []
    print(edges)
    for ed in edges:
        if ed[0] == 0 and ed[1] == L-1:
            cust_edges.append(ed[::-1])
        else:
            cust_edges.append(ed)
    return GraphCust(n_nodes=L, edges=cust_edges)

def get_cluster_graph_J2(L, PBC):
    '''
    create the GraphCust (custom graph object) for a J1-J2 chain with
    clusters as sites.

    *** Currently only works for b=2 ***
    '''
    graph1 = nk.graph.Hypercube(length=L, n_dim=1, pbc=PBC)
    edges = graph1.edges()
    
    cust_edges = []
    for ed in edges:
        if ed[0] == 0 and ed[1] == L-1:
            cust_edges.append((*ed[::-1], 0))  # J1, color 0
        else:
            cust_edges.append((*ed, 0))         # J1, color 0
    
    # J2 edges: same connectivity, color 1
    for ed in edges:
        if ed[0] == 0 and ed[1] == L-1:
            cust_edges.append((*ed[::-1], 1))  # J2, color 1
        else:
            cust_edges.append((*ed, 1))         # J2, color 1

    return GraphCust(n_nodes=L, edges=cust_edges)



class Heisenberg_cluster_Alex(GraphOperatorCust):
    r"""
    New version of heisenberg cluster basis hamiltonian

    Now should take site_ops and bond_ops each as a list 

    J1, J2 don't need to be passed b/c they are contained in 
    site_ops and bond_ops

    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        n: int,
        graph: AbstractGraph,
        Nlocal: int,
        site_ops: list,
        bond_ops: list, # should be [bondops_J1, bondops_J2]
        J: float | Sequence[float] = 1.0,
        sign_rule: None | bool | Sequence[bool] = None,
        dtype: DType | None = None,
        *,
        acting_on_subspace: None | list[int] | int = None,
    ):

  
        self._J = J
        self._sign_rule = sign_rule
        self.n = n

        self.site_ops = site_ops
        self.bond_ops = bond_ops

        bond_ops_colors = [0, 1]

        

        super().__init__(
            hilbert,
            graph,
            bond_ops=bond_ops,
            site_ops=site_ops,
            bond_ops_colors=bond_ops_colors,
            dtype=dtype,
            acting_on_subspace=acting_on_subspace,
        )

    @property
    def J(self) -> float:
        """The coupling strength."""
        return self._J

    @property
    def uses_sign_rule(self):
        return self._sign_rule

    def __repr__(self):
        return f"Heisenberg(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"
