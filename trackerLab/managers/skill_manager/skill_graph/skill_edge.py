import torch
from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Union, List, Literal
import networkx as nx
from .skill_data import SkillData, MotionLib

@configclass
class SkillEdge:
    from_node: int = MISSING
    to_node: int = MISSING
    is_default_edge: bool = False
    blend: int = 0
    blend_type: str = "linear"

def compute_state_distance(
    state1: torch.Tensor, state2: torch.Tensor,
    norm: str = "l2",  metric: str = "frame", window: int = 5
) -> float:
    """
    Compute distance or dissimilarity between two states or sequences.
    Supports l1, l2, and cosine similarity. Optionally uses a temporal window.
    """
    if metric == "frame":
        if norm == "l2":
            return torch.norm(state1 - state2, p=2).item()
        elif norm == "l1":
            return torch.norm(state1 - state2, p=1).item()
        elif norm == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(
                state1.unsqueeze(0), state2.unsqueeze(0)
            ).item()
            return 1 - cos_sim
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    elif metric == "window":
        assert state1.ndim == 2 and state2.ndim == 2, "Window mode requires sequences"
        assert state1.shape[0] >= window and state2.shape[0] >= window
        slice1 = state1[-window:]
        slice2 = state2[:window]
        if norm == "l2":
            return torch.norm(slice1 - slice2, p=2).item()
        elif norm == "l1":
            return torch.norm(slice1 - slice2, p=1).item()
        elif norm == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(
                slice1.flatten().unsqueeze(0), slice2.flatten().unsqueeze(0)
            ).item()
            return 1 - cos_sim
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def check_adjecent(
    data1: SkillData, data2: SkillData, mlib: MotionLib, comp: str = "dof_pos",
    threshold: float = 0.5, norm: str = "l2", metric: Literal["frame", "window"] = "frame", window: int = 5
) -> bool:
    """
    Check whether two skill segments can be connected.
    Uses distance between terminal state of data1 and starting state of data2.
    """
    if metric == "frame":
        end_state_1 = data1.get_terminal_state(mlib, comp)
        start_state_2 = data2.get_start_state(mlib, comp)
    elif metric == "window":
        end_state_1 = data1.get_data(mlib, comp)
        start_state_2 = data2.get_data(mlib, comp)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    dist = compute_state_distance(
        end_state_1, start_state_2, norm=norm, metric=metric, window=window
    )
    return dist < threshold

def build_skill_edges(
    skill_list: List[SkillData], mlib: MotionLib, threshold: float = 0.5, comp: str = "dof_pos",
    norm: str = "l2", metric: Literal["frame", "window"] = "frame", window: int = 5
) -> List[SkillEdge]:
    """
    Build a list of directed skill edges based on adjacency criteria.
    """
    edges = []
    print("Building skill edges...")
    for i in range(len(skill_list)):
        for j in range(len(skill_list)):
            if check_adjecent(
                skill_list[i], skill_list[j], mlib,
                comp=comp, threshold=threshold, norm=norm,
                metric=metric, window=window
            ):
                edges.append(SkillEdge(from_node=i, to_node=j, is_default_edge=False))
    print(f"Total edges created: {len(edges)}")
    return edges

def build_skill_edges_all(
    skill_list: List[SkillData], mlib: MotionLib, threshold: float = 0.5, comp: str = "dof_pos",
    norm: str = "l2", metric: Literal["frame", "window"] = "frame", window: int = 5
) -> List[SkillEdge]:
    edges = []
    print("Building skill edges...")
    for i in range(len(skill_list)):
        for j in range(len(skill_list)):
            edges.append(SkillEdge(from_node=i, to_node=j, is_default_edge=True))
    print(f"Total edges created: {len(edges)}")
    return edges

def build_skill_edges_connected(
    skill_list: List[SkillData],
    mlib: MotionLib,
    threshold: float = 0.5,
    comp: str = "dof_pos",
    norm: str = "l2",
    metric: Literal["frame", "window"] = "frame",
    window: int = 5
) -> List[SkillEdge]:
    """
    Build a list of directed skill edges based on adjacency criteria.
    Ensures that each node has at least one incoming and one outgoing edge.
    """
    N = len(skill_list)
    edges: List[SkillEdge] = []
    edge_set = set()

    # Distance matrix
    dist_matrix = torch.full((N, N), float("inf"))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            data_i = skill_list[i]
            data_j = skill_list[j]
            dist = compute_state_distance(
                data_i.get_terminal_state(mlib, comp) if metric == "frame" else data_i.get_data(mlib, comp),
                data_j.get_start_state(mlib, comp) if metric == "frame" else data_j.get_data(mlib, comp),
                norm=norm,
                metric=metric,
                window=window,
            )
            dist_matrix[i, j] = dist
            if dist < threshold:
                edges.append(SkillEdge(from_node=i, to_node=j, is_default_edge=False))
                edge_set.add((i, j))

    # Ensure out edges
    for i in range(N):
        min_j = torch.argmin(dist_matrix[i]).item()
        if (i, min_j) not in edge_set:
            edges.append(SkillEdge(from_node=i, to_node=min_j, is_default_edge=True))
            edge_set.add((i, min_j))
        else:
            for e in edges:
                if e.from_node == i and e.to_node == min_j:
                    e.is_default_edge = True
                    break

    # Ensure in edges
    for j in range(N):
        if not any(e.to_node == j for e in edges):
            # find the closest node that connects to j
            min_i = torch.argmin(dist_matrix[:, j]).item()
            if (min_i, j) not in edge_set:
                edges.append(SkillEdge(from_node=min_i, to_node=j, is_default_edge=False))
                edge_set.add((min_i, j))

    print(f"Total edges created: {len(edges)}")
    return edges


def build_skill_edges_strong(
    skill_list: List[SkillData],
    mlib: MotionLib,
    threshold: float = 0.5,
    comp: str = "dof_pos",
    norm: str = "l2",
    metric: Literal["frame", "window"] = "frame",
    window: int = 5,
) -> List[SkillEdge]:
    """
    Build a strongly connected skill graph by connecting adjacent skill segments.
    """
    N = len(skill_list)
    edges: List[SkillEdge] = []
    edge_set = set()

    # Distance matrix
    dist_matrix = torch.full((N, N), float("inf"))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            state_i = (
                skill_list[i].get_terminal_state(mlib, comp)
                if metric == "frame"
                else skill_list[i].get_data(mlib, comp)[-window:]
            )
            state_j = (
                skill_list[j].get_start_state(mlib, comp)
                if metric == "frame"
                else skill_list[j].get_data(mlib, comp)[:window]
            )
            dist = compute_state_distance(state_i, state_j, norm=norm, metric="frame")
            dist_matrix[i, j] = dist
            if dist < threshold:
                edges.append(SkillEdge(from_node=i, to_node=j, is_default_edge=False))
                edge_set.add((i, j))

    # Create directed graph for connectivity analysis
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_set)

    # Check strongly connected components
    scc = list(nx.strongly_connected_components(G))
    if len(scc) > 1:
        print(f"Graph is not strongly connected, has {len(scc)} components. Fixing...")

        # Convert to list of components
        comp_list = [list(c) for c in scc]

        # Connect components linearly to form strong connectivity
        for k in range(len(comp_list) - 1):
            from_group = comp_list[k]
            to_group = comp_list[k + 1]

            # Find min-distance pair across groups
            min_dist = float("inf")
            best_pair = (None, None)
            for i in from_group:
                for j in to_group:
                    d = dist_matrix[i, j]
                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)
            i, j = best_pair
            if (i, j) not in edge_set:
                edges.append(SkillEdge(from_node=i, to_node=j, is_default_edge=True))
                edge_set.add((i, j))
                G.add_edge(i, j)

        # Ensure final loop closure to make it cyclic
        if len(comp_list) > 1:
            first = comp_list[0][0]
            last = comp_list[-1][0]
            if (last, first) not in edge_set:
                edges.append(SkillEdge(from_node=last, to_node=first, is_default_edge=True))
                edge_set.add((last, first))

    print(f"Total edges created: {len(edges)}")
    return edges

BUILD_GRAPH_METHODS = {
    "build_skill_edges": build_skill_edges, 
    "build_skill_edges_connected": build_skill_edges_connected, 
    "build_skill_edges_strong": build_skill_edges_strong, 
    "build_skill_edges_all": build_skill_edges_all
}
