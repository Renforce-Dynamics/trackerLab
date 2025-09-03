import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Union
from ..import SkillGraph, SkillData, SkillEdge



class SkillGraphVisualizer:
    def __init__(self, skill_list: List['SkillData'], skill_edges: List['SkillEdge']):
        self.skill_list = skill_list
        self.skill_edges = skill_edges
        self.skill_to_patch = self._build_skill_to_patch()
        self.patch_to_skill = self._build_patch_to_skill()
        self.patch_edges = self._build_patch_edges()
    
    @classmethod
    def from_graph(cls, graph:SkillGraph):
        return cls(graph.skill_list, graph.skill_edges)
    
    def _build_skill_to_patch(self):
        # [[0,1], [2,3,4], ...]
        skill_to_patch = []
        counter = 0
        for skill in self.skill_list:
            patches = list(range(counter, counter + skill.num_patches))
            skill_to_patch.append(patches)
            counter += skill.num_patches
        return skill_to_patch

    def _build_patch_to_skill(self):
        mapping = {}
        for skill_id, patches in enumerate(self.skill_to_patch):
            for patch_id in patches:
                mapping[patch_id] = skill_id
        return mapping

    def _build_patch_edges(self):
        edges = []
        for patches in self.skill_to_patch:
            for i in range(len(patches) - 1):
                edges.append(SkillEdge(from_node=patches[i], to_node=patches[i+1]))
        for edge in self.skill_edges:
            from_skill_patches = self.skill_to_patch[edge.from_node]
            to_skill_patches = self.skill_to_patch[edge.to_node]
            if from_skill_patches and to_skill_patches:
                edges.append(SkillEdge(
                    from_node=from_skill_patches[-1],
                    to_node=to_skill_patches[0],
                    is_default_edge=edge.is_default_edge,
                    blend=edge.blend,
                    blend_type=edge.blend_type
                ))
        return edges

    def draw_skill_graph(self, ax, title="Skill-Level Graph"):
        G = nx.DiGraph()
        for skill_id, skill in enumerate(self.skill_list):
            label = skill.desc or skill.name or f"Skill {skill_id}"
            G.add_node(f"s{skill_id}", label=label)
        for edge in self.skill_edges:
            style = 'dotted' if edge.is_default_edge else 'solid'
            G.add_edge(f"s{edge.from_node}", f"s{edge.to_node}", style=style)

        pos = nx.spring_layout(G, seed=42)
        self._draw_graph(G, pos, ax, title)

    def draw_patch_graph(self, ax, title="Patch-Level Graph"):
        G = nx.DiGraph()
        for patches in self.skill_to_patch:
            for patch_id in patches:
                G.add_node(f"p{patch_id}", label=f"p{patch_id}")
        for edge in self.patch_edges:
            style = 'dotted' if edge.is_default_edge else 'solid'
            G.add_edge(f"p{edge.from_node}", f"p{edge.to_node}", style=style)

        pos = nx.spring_layout(G, seed=42)
        self._draw_graph(G, pos, ax, title)

    def draw_skill_patch_map(self, ax, title="Skill-to-Patch Mapping"):
        G = nx.DiGraph()
        for skill_id, patches in enumerate(self.skill_to_patch):
            skill_node = f"s{skill_id}"
            label = self.skill_list[skill_id].desc  or self.skill_list[skill_id].name or f"Skill {skill_id}"
            G.add_node(skill_node, label=label)
            for patch_id in patches:
                patch_node = f"p{patch_id}"
                G.add_node(patch_node, label=patch_node)
                G.add_edge(skill_node, patch_node, style='dashed')

        pos = nx.spring_layout(G, seed=42)
        self._draw_graph(G, pos, ax, title)

    def _draw_graph(self, G, pos, ax, title):
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
        dotted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dotted"]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=600)
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9)
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, ax=ax, edge_color='gray')
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, ax=ax, style='dashed', edge_color='blue')
        nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, ax=ax, style='dotted', edge_color='red')

        ax.set_title(title)
        ax.axis('off')