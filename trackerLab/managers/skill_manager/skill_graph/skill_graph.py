import torch
from collections import defaultdict
import random

from typing import Union, List, Literal, Dict
from isaaclab.utils import configclass
from .skill_data import SkillData
from .skill_edge import SkillEdge

from dataclasses import MISSING

@configclass
class SkillGraphCfg:
    patch_size: int = 10
    skill_list: Union[List[SkillData], str] = MISSING
    skill_edges: Union[List[SkillEdge], str] = MISSING

MAX_WEIGHT = 1e9

class SkillGraph(object):
    def __init__(self, cfg: SkillGraphCfg, device):
        self.cfg = cfg
        self.device = device
        self.skill_list = cfg.skill_list
        self.skill_edges = cfg.skill_edges
        self.num_skills = len(self.skill_list)
        self.init_graph()

    def init_graph(self):
        self.init_skill_graph()
        self.init_patch_graph()

    def init_skill_graph(self):
        skill_start_patch_ids = []
        acc = 0
        for skill in self.skill_list:
            skill_start_patch_ids.append(acc)
            acc += skill.num_patches
        self.skill_start_patch_ids = torch.tensor(skill_start_patch_ids, dtype=torch.long, device=self.device)
        self.num_patch = acc
        
        self.skill_graph = self.build_skill_graph(self.skill_list, self.skill_edges)
        self.default_skill_trans = self.build_default_skill_transitions(self.skill_list, self.skill_edges)
        
    def init_patch_graph(self):
        patch_to_skill = []
        for i, skill in enumerate(self.skill_list):
            patch_to_skill += [i] * skill.num_patches
        # patch_to_skill = torch.tensor(patch_to_skill, device=self.device)
        self.patch_to_skill = torch.tensor(patch_to_skill, dtype=torch.long)
        
        self.patch_graph = self.build_skill_patch_graph(self.skill_list, self.skill_edges)
        self.default_patch_trans = self.build_default_patch_transitions(self.skill_list, self.skill_edges)

    def is_valid_transition_batch(self, curr_patch_ids: torch.LongTensor, next_patch_ids: torch.LongTensor) -> torch.BoolTensor:
        costs = self.patch_graph[curr_patch_ids, next_patch_ids]  # [B]
        return costs < MAX_WEIGHT

    def transition_batch(self, curr_patch_ids: torch.LongTensor, next_patch_ids: torch.LongTensor):
        valid_mask = self.is_valid_transition_batch(curr_patch_ids, next_patch_ids)  # [B]
        fallback_patch_ids = self.default_patch_trans[curr_patch_ids].long()
        target_patch_ids = torch.where(valid_mask, next_patch_ids, fallback_patch_ids)
        costs = self.patch_graph[curr_patch_ids, target_patch_ids]
        return target_patch_ids, costs

    def skill_to_patch_start(self, skill_ids: torch.LongTensor) -> torch.LongTensor:
        return self.skill_start_patch_ids[skill_ids]

    def step_from_skill_id(self, curr_patch_ids: torch.LongTensor, skill_ids: torch.LongTensor):
        target_patch_ids = self.skill_to_patch_start(skill_ids.to(self.device))
        next_patch_ids, costs = self.transition_batch(curr_patch_ids, target_patch_ids)
        return next_patch_ids, costs

    def default_transition_from_patch(self, curr_patch_ids: torch.LongTensor):
        return self.default_patch_trans[curr_patch_ids]

    def random_transition_from_patch(self, curr_patch_ids: torch.LongTensor):
        legal_mask = self.patch_graph[curr_patch_ids] < MAX_WEIGHT  # [B, P]
        # Assert that there must be at least one valid transition
        assert legal_mask.sum(dim=-1).min() > 0, "No valid transition found for the current patch IDs."
        probs = legal_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True)  # [B, P], row-wise normalize
        sampled_patch_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        costs = self.patch_graph[curr_patch_ids, sampled_patch_ids]
        return sampled_patch_ids, costs

    def get_action_mask(self, curr_patch_ids: torch.LongTensor) -> torch.BoolTensor:
        skill_start_patch_ids = torch.tensor(self.skill_start_patch_ids, device=self.device)  # [S]

        # mask: [B, S]
        B = curr_patch_ids.shape[0]
        S = len(skill_start_patch_ids)
        curr_patch_ids_expand = curr_patch_ids.unsqueeze(1).expand(-1, S)  # [B, S]
        target_patch_ids_expand = skill_start_patch_ids.unsqueeze(0).expand(B, -1)  # [B, S]
        
        legal_mask = self.is_valid_transition_batch(curr_patch_ids_expand, target_patch_ids_expand)  # [B, S]
        return legal_mask  # shape: [B, num_skills]

    def build_skill_graph(self, skill_list: list[SkillData], edges: list[SkillEdge]):
        """
        Build skill level graph
        """
        N = len(skill_list)
        W = torch.full((N, N), MAX_WEIGHT, device=self.device)
        for edge in edges:
            W[edge.from_node, edge.to_node] = edge.blend
        return W
    
    def build_skill_patch_graph(self, skill_list: list[SkillData], edges: list[SkillEdge]):
        """
        Build patch-level graph: each skill is a chain of patches;
        transitions between skills are edges between last patch of skill_i to first patch of skill_j
        """
        total_patches = sum(skill.num_patches for skill in skill_list)
        W = torch.full((total_patches, total_patches), MAX_WEIGHT, device=self.device)

        offset = 0
        for skill in skill_list:
            for i in range(skill.num_patches - 1):
                W[offset + i, offset + i + 1] = 0
            offset += skill.num_patches

        skill_patch_start = 0
        skill_patch_starts = []
        for skill in skill_list:
            skill_patch_starts.append(skill_patch_start)
            skill_patch_start += skill.num_patches

        for edge in edges:
            from_last_patch = skill_patch_starts[edge.from_node] + skill_list[edge.from_node].num_patches - 1
            to_first_patch = skill_patch_starts[edge.to_node]
            W[from_last_patch, to_first_patch] = edge.blend

        return W
    
    def build_default_skill_transitions(self, skill_list: list[SkillData], edges: list[SkillEdge]):
        total_skills = len(skill_list)
        default_transitions = torch.arange(total_skills, device=self.device)
        
        for edge in edges:
            if edge.is_default_edge:
                default_transitions[edge.from_node] = edge.to_node
        return default_transitions
    
    def build_default_patch_transitions(self, skill_list: list[SkillData], edges: list[SkillEdge]):
        total_patches = sum(skill.num_patches for skill in skill_list)
        default_transitions = torch.arange(total_patches, device=self.device)

        patch_offset = 0
        for skill_idx, skill in enumerate(skill_list):
            for i in range(skill.num_patches):
                patch_idx = patch_offset + i
                default_transitions[patch_idx] = patch_idx + 1
            # For end patch it requires explicit define, or stuck in current
            default_transitions[patch_idx] = patch_idx
            patch_offset += skill.num_patches

        skill_patch_start = 0
        skill_patch_starts = []
        for skill in skill_list:
            skill_patch_starts.append(skill_patch_start)
            skill_patch_start += skill.num_patches

        for edge in edges:
            if edge.is_default_edge:
                from_last_patch = skill_patch_starts[edge.from_node] + skill_list[edge.from_node].num_patches - 1
                to_first_patch = skill_patch_starts[edge.to_node]
                default_transitions[from_last_patch] = to_first_patch

        return default_transitions
