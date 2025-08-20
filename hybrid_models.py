# Copyright (c) Meta Platforms, Inc. and affiliates.
# Hybrid CDiT model integrating WorldMem memory mechanisms
# 
# This implementation combines:
# - CDiT's precise conditional guidance through cross-attention
# - WorldMem's long-term memory capabilities through selective memory access
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Union, List
from timm.layers.patch_embed import PatchEmbed
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp
from models import TimestepEmbedder, ActionEmbedder, modulate, FinalLayer


class ZeroParameterAdaptiveScoring:
    """
    Zero-parameter adaptive scoring system (Stage 2 optimization)
    Dynamically adjust retrieval weights, does not affect storage logic, no training required
    """
    
    def compute_adaptive_weights(self, current_situation):
        """
        Dynamically adjust retrieval weights based on current situation, no training required
        
        Args:
            current_situation: dict containing {'action': tensor, 'pose': tensor}
            
        Returns:
            dict: Adaptively adjusted retrieval weights
        """
        action = current_situation['action']
        
        # 1. Weight adjustment based on linear motion magnitude
        linear_magnitude = torch.norm(action[:2]).item()
        
        if linear_magnitude < 0.1:  # Stationary/slow movement
            action_weight = 0.2      # Reduce action weight, as action is not obvious
            spatial_weight = 0.6     # Significantly increase spatial weight, rely on position info
            memory_weight = 0.15     # Moderate memory weight
            usage_weight = 0.05      # Low usage weight
        elif linear_magnitude > 0.4:  # Fast movement
            action_weight = 0.45     # Moderately increase action weight, but not exceeding spatial
            spatial_weight = 0.35    # Keep spatial weight as dominant
            memory_weight = 0.15     # Moderate memory weight
            usage_weight = 0.05      # Low usage weight
        else:  # Medium speed (0.1 <= magnitude <= 0.4)
            action_weight = 0.35     # Balanced weight, but spatial priority
            spatial_weight = 0.45    # Spatial weight still dominant
            memory_weight = 0.15     # Moderate memory weight
            usage_weight = 0.05      # Low usage weight
            
        # 2. Further adjustment based on turn magnitude
        turn_magnitude = abs(action[2]).item()
        
        if turn_magnitude > 0.3:  # Large turn (above ~17°)
            action_weight = min(action_weight + 0.15, 0.6)  # Moderately increase action weight, but not exceeding spatial
            spatial_weight = max(spatial_weight - 0.1, 0.25)  # Slightly reduce spatial weight, but maintain significant position
        elif turn_magnitude > 0.15:  # Medium turn
            action_weight = min(action_weight + 0.08, 0.5)  # Slightly increase action weight
            spatial_weight = max(spatial_weight - 0.05, 0.3)  # Slightly reduce spatial weight
            
        # 3. Ensure weights sum to 1.0
        total_weight = action_weight + spatial_weight + memory_weight + usage_weight
        action_weight /= total_weight
        spatial_weight /= total_weight
        memory_weight /= total_weight
        usage_weight /= total_weight
        
        return {
            'retrieval_action_weight': action_weight,
            'retrieval_spatial_weight': spatial_weight,
            'retrieval_memory_weight': memory_weight,
            'retrieval_usage_weight': usage_weight
        }
    
    def compute_situation_complexity(self, current_situation):
        """
        Compute complexity of current situation to decide whether to use adaptive weights
        
        Returns:
            float: Complexity score (0-1), higher means more complex, more need for adaptive adjustment
        """
        action = current_situation['action']
        
        linear_magnitude = torch.norm(action[:2]).item()
        turn_magnitude = abs(action[2]).item()
        
        # Complexity scoring: very slow, very fast, large turns are considered complex situations
        complexity = 0.0
        
        # Linear motion complexity
        if linear_magnitude < 0.05:  # Very slow
            complexity += 0.3
        elif linear_magnitude > 0.5:  # Very fast
            complexity += 0.4
        
        # Turn complexity
        if turn_magnitude > 0.4:  # Large turn
            complexity += 0.5
        elif turn_magnitude > 0.2:  # Medium turn
            complexity += 0.3
            
        return min(complexity, 1.0)


class MemoryBuffer:
    """
    Intelligent Memory Buffer System - LT-NWM Core Innovation
    
    Multi-factor Scored Storage based on:
    • Turning behavior (35%) - primary behavioral indicator
    • Spatial uniqueness (30%) - spatial distribution optimization
    • Maneuver complexity (15%) - complex scenario prioritization
    
    Dynamic Decay Mechanism:
    • Time-based decay for automatic low-value memory phase-out
    • Usage-based score boosting for frequently accessed memories
    
    Adaptive Weight Adjustment:
    • Dynamically adjusts retrieval weights based on scene complexity
    """
    def __init__(self, max_size: int = 40):
        # === Configurable scoring parameters (Stage 1: simplified scoring based on turns and space) ===
        self.SCORING_CONFIG = {
            # === Storage standard parameters (Stage 1: focus on turning behavior) ===
            'storage_turn_weight': 35.0,         # Storage: turning action importance
            'storage_sharp_turn_weight': 50.0,   # Storage: sharp turn extra weighting
            'storage_spatial_weight': 30.0,      # Storage: spatial uniqueness weight
            'storage_complex_maneuver': 15.0,    # Storage: complex maneuver bonus
            'storage_trivial_penalty': -8.0,     # Storage: trivial action penalty
            'storage_close_penalty': -12.0,      # Storage: position too close penalty
            'storage_first_frame_bonus': 40.0,   # Storage: first frame starting point bonus
            'storage_min_distance': 4.0,         # Storage: minimum spatial distance requirement
            
            # === Retrieval standard parameters (flexible matching of relevant memories) ===
            'retrieval_action_weight': 0.35,     # Retrieval: action similarity weight (important)
            'retrieval_memory_weight': 0.20,     # Retrieval: memory value weight (important)
            'retrieval_spatial_weight': 0.40,    # Retrieval: spatial relevance weight (primary) - elevated to dominant position
            'retrieval_usage_weight': 0.05,      # Retrieval: usage experience weight (auxiliary)
            'retrieval_spatial_radius': 10.0,    # Retrieval: spatial matching radius
            
            # === Dynamic decay system parameters ===
            'usage_boost': 5.0,                  # Score boost per usage
            'fixed_memory_time': 3,               # Fixed memory time tx: no decay for first 3 steps
            'base_decay_rate': 1.2,               # Base decay rate
            'accelerated_decay_rate': 1.4,        # Accelerated decay rate a: decay speed increment coefficient
            'max_score': 100.0,                  # Maximum score limit
            'min_survival_score': 3.0,           # Minimum retention score
            
            # === Behavior classification thresholds (Stage 1: significant turns and sharp turns) ===
            'significant_turn_threshold': 0.25,  # Significant turn threshold (~14°)
            'sharp_turn_threshold': 0.45,        # Sharp turn threshold (~26°)
            'linear_motion_threshold': 0.2,      # Linear motion threshold
            
            # === Normalized thresholds (for potential future use) ===
            'norm_significant_turn_threshold': 0.15,  # Normalized significant turn threshold
            'norm_sharp_turn_threshold': 0.3,         # Normalized sharp turn threshold
            'norm_linear_motion_threshold': 0.1,      # Normalized linear motion threshold
        }
        
        self.max_size = max_size
        
        # Stage 2 optimization: zero-parameter adaptive scoring system
        self.adaptive_scorer = ZeroParameterAdaptiveScoring()
        
        # Storage structure
        self.frames = []
        self.poses = []
        self.actions = []
        self.frame_indices = []
        self.scores = []           # Current score for each memory (0-100 points)
        self.usage_counts = []     # Usage count statistics
        self.last_used = []        # Last usage time
        self.unused_steps = []     # Consecutive unused steps (key metric for dynamic decay)
        self.current_frame_idx = 0 # Current frame index for decay calculations
        
    def add_frame(self, frame_latent: torch.Tensor, pose: torch.Tensor, action: Optional[torch.Tensor] = None, frame_idx: int = 0):
        """Intelligently add frame to memory cache, always retain the highest-scored 40 frames"""
        # Calculate new frame's storage value score
        storage_score = self.compute_storage_score(pose, action, frame_idx)
        
        # If cache is not full, add directly
        if len(self.frames) < self.max_size:
            self.frames.append(frame_latent.detach())
            self.poses.append(pose.detach())
            if action is not None:
                self.actions.append(action.detach())
            else:
                self.actions.append(torch.zeros(3, device=frame_latent.device))
            self.frame_indices.append(frame_idx)
            self.scores.append(storage_score)
            self.usage_counts.append(0)
            self.last_used.append(frame_idx)
            self.unused_steps.append(0)  # Initialize new memory with 0 unused steps
            return True
        
        # Cache is full, need to replace the lowest-scoring memory
        min_score_idx = self.scores.index(min(self.scores))
        min_score = self.scores[min_score_idx]
        
        # If new frame score is higher, replace the lowest-scoring memory
        if storage_score > min_score:
            self.frames[min_score_idx] = frame_latent.detach()
            self.poses[min_score_idx] = pose.detach()
            if action is not None:
                self.actions[min_score_idx] = action.detach()
            else:
                self.actions[min_score_idx] = torch.zeros(3, device=frame_latent.device)
            self.frame_indices[min_score_idx] = frame_idx
            self.scores[min_score_idx] = storage_score
            self.usage_counts[min_score_idx] = 0
            self.last_used[min_score_idx] = frame_idx
            self.unused_steps[min_score_idx] = 0  # Reset new memory unused steps
            return True
        
        return False
    
    def compute_storage_score(self, pose: torch.Tensor, action: Optional[torch.Tensor] = None, frame_idx: int = 0) -> float:
        """
        Calculate frame storage value score (Stage 1: simplified scoring based on turns and space)
        Focus: turning behavior detection + spatial uniqueness
        
        Args:
            pose: Current position [x, y, z, yaw]
            action: Current action [delta_x, delta_y, delta_yaw] (normalized)
            frame_idx: Frame index
            
        Returns:
            float: Storage value score (0-100 points)
        """
        score = 0.0
        config = self.SCORING_CONFIG
        
        # 1. Turning behavior detection (Stage 1 focus: key action recognition)
        if action is not None:
            turn_magnitude = torch.abs(action[2]).item()  # |delta_yaw|
            linear_magnitude = torch.norm(action[:2]).item()
            
            # Sharp turn action - important navigation nodes
            if turn_magnitude >= config['sharp_turn_threshold']:
                score += config['storage_sharp_turn_weight']  # +50 points: sharp turn landmarks
            elif turn_magnitude >= config['significant_turn_threshold']:
                score += config['storage_turn_weight']  # +35 points: significant turns
            
            # Complex maneuver - turning while moving forward
            if turn_magnitude >= 0.15 and linear_magnitude >= config['linear_motion_threshold']:
                score += config['storage_complex_maneuver']  # +15 points: complex maneuvers
            
            # Trivial straight-line action penalty
            if turn_magnitude < 0.1 and linear_magnitude < 0.15:
                score += config['storage_trivial_penalty']  # -8 points: trivial actions
        
        # 2. Spatial uniqueness detection (Stage 1 focus: avoid repeated positions)
        if len(self.poses) > 0:
            current_pos = pose[:3]
            stored_poses = torch.stack(self.poses)
            distances = torch.norm(stored_poses[:, :3] - current_pos, dim=1)
            min_distance = torch.min(distances).item()
            
            # New area exploration bonus
            if min_distance >= config['storage_min_distance']:
                # Greater distance, higher storage value
                distance_score = config['storage_spatial_weight'] * min(min_distance / 8.0, 2.0)
                score += distance_score  # Up to +60 points: new areas
            else:
                # Too close distance penalty
                score += config['storage_close_penalty']  # -12 points: repeated positions
        else:
            # First frame starting point bonus
            score += config['storage_first_frame_bonus']  # +40 points: important starting point
        
        # Limit score range
        final_score = min(max(score, 0.0), config['max_score'])
        return final_score
    
    def compute_retrieval_score(self, current_pose: torch.Tensor, target_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate retrieval relevance scores to determine which memories are most helpful for current inference
        Stage 2 optimization: use zero-parameter adaptive weight system
        
        Args:
            current_pose: Current position
            target_action: Target action (action to be executed currently)
            
        Returns:
            torch.Tensor: Retrieval relevance score for each memory
        """
        if len(self.frames) == 0:
            return torch.tensor([])
        
        device = current_pose.device
        
        # Core strategy: similar actions should produce similar changes
        if target_action is not None:
            # Stage 2 optimization: compute adaptive weights
            current_situation = {
                'action': target_action.to(device),
                'pose': current_pose.to(device)
            }
            complexity = self.adaptive_scorer.compute_situation_complexity(current_situation)
            
            # If situation complexity is high, use adaptive weights; otherwise use default weights
            if complexity > 0.3:  # Complex situation threshold
                adaptive_weights = self.adaptive_scorer.compute_adaptive_weights(current_situation)
                action_weight = adaptive_weights['retrieval_action_weight']
                memory_weight = adaptive_weights['retrieval_memory_weight']
                spatial_weight = adaptive_weights['retrieval_spatial_weight']
                usage_weight = adaptive_weights['retrieval_usage_weight']
            else:
                # Use default weights
                config = self.SCORING_CONFIG
                action_weight = config['retrieval_action_weight']
                memory_weight = config['retrieval_memory_weight']
                spatial_weight = config['retrieval_spatial_weight']
                usage_weight = config['retrieval_usage_weight']
            
            # 1. Action behavioral similarity (primary retrieval factor)
            memory_actions = torch.stack(self.actions).to(device)
            action_similarities = self._compute_action_similarity_for_retrieval(
                target_action.to(device), memory_actions
            )
            
            # 2. Memory storage value weighting (secondary factor)
            memory_scores = torch.tensor(self.scores, device=device)
            # Normalize scores to [0,1]
            norm_scores = memory_scores / self.SCORING_CONFIG['max_score']
            
            # 3. Spatial context relevance (auxiliary factor)
            current_pos = current_pose[:3]
            memory_poses = torch.stack(self.poses).to(device)
            spatial_dists = torch.norm(memory_poses[:, :3] - current_pos, dim=1)
            spatial_similarities = torch.exp(-spatial_dists / self.SCORING_CONFIG['retrieval_spatial_radius'])  # Use retrieval-specific radius
            
            # 4. Usage frequency weighting (experience factor)
            usage_scores = torch.tensor(self.usage_counts, device=device, dtype=torch.float)
            usage_weights = torch.log(usage_scores + 1) / 5.0  # Logarithmic scaling, avoid excessive bias
            
            # Comprehensive retrieval scoring: use adaptive weights
            retrieval_scores = (action_weight * action_similarities + 
                               memory_weight * norm_scores + 
                               spatial_weight * spatial_similarities +
                               usage_weight * usage_weights)
        else:
            # When no target action, primarily based on memory value and spatial relevance
            memory_scores = torch.tensor(self.scores, device=device)
            norm_scores = memory_scores / self.SCORING_CONFIG['max_score']
            
            current_pos = current_pose[:3]
            memory_poses = torch.stack(self.poses).to(device)
            spatial_dists = torch.norm(memory_poses[:, :3] - current_pos, dim=1)
            spatial_similarities = torch.exp(-spatial_dists / 8.0)
            
            usage_scores = torch.tensor(self.usage_counts, device=device, dtype=torch.float)
            usage_weights = torch.log(usage_scores + 1) / 5.0
            
            retrieval_scores = 0.5 * norm_scores + 0.3 * spatial_similarities + 0.2 * usage_weights
        
        return retrieval_scores
    
    def update_usage_scores(self, used_indices: List[int], current_frame_idx: int):
        """
        Dynamic decay system: implement fixed memory time + accelerated decay mechanism
        
        Design logic:
        1. Fixed memory time tx=3 steps: no decay for first 3 steps
        2. Start decay from 4th step: decay speed increases with unused steps
        3. Boost score and reset decay cooldown when used
        
        Args:
            used_indices: List of memory indices used in current inference
            current_frame_idx: Current frame index
        """
        config = self.SCORING_CONFIG
        fixed_memory_time = config['fixed_memory_time']      # tx = 3 steps fixed memory time
        base_decay = config['base_decay_rate']               # Base decay rate
        accel_decay = config['accelerated_decay_rate']       # Accelerated decay rate a
        
        # 1. Boost used memory scores and reset decay cooldown
        for idx in used_indices:
            if 0 <= idx < len(self.scores):
                # Score boost
                self.scores[idx] = min(
                    self.scores[idx] + config['usage_boost'],
                    config['max_score']
                )
                self.usage_counts[idx] += 1
                self.last_used[idx] = current_frame_idx
                # Key: reset decay cooldown period, enjoy 3-step protection again
                self.unused_steps[idx] = 0
        
        # 2. Apply dynamic decay system to unused memories
        for i in range(len(self.scores)):
            if i not in used_indices:
                # Increase consecutive unused steps
                self.unused_steps[i] += 1
                
                # Fixed memory time protection: no decay for first tx steps
                if self.unused_steps[i] <= fixed_memory_time:
                    continue  # Skip decay, enjoy protection period
                
                # Start dynamic decay: from 4th step onwards
                excess_steps = self.unused_steps[i] - fixed_memory_time  # Steps exceeding protection period
                
                # Accelerated decay formula: decay rate increases with unused steps
                # decay = base_decay * (accel_decay ^ excess_steps)
                # This way 4th step decays slowly, but gets faster later
                dynamic_decay_rate = base_decay * (accel_decay ** excess_steps)
                
                # Apply decay
                self.scores[i] = max(
                    self.scores[i] - dynamic_decay_rate,
                    config['min_survival_score']
                )
                
                # Debug info: can be enabled when needed
                # print(f"Memory {i}: unused_steps={self.unused_steps[i]}, "
                #       f"excess_steps={excess_steps}, dynamic_decay={dynamic_decay_rate:.2f}, "
                #       f"new_score={self.scores[i]:.2f}")
    
    
    def should_store_frame(self, pose: torch.Tensor, action: Optional[torch.Tensor] = None, 
                          frame_idx: int = 0, min_distance: float = 5.0) -> bool:
        """
        Determine whether current frame should be stored in memory buffer
        Stage 1 strategy: always calculate score, retain highest 40 frames
        """
        # Always return True, let add_frame method handle replacement logic
        return True
    
    def get_relevant_frames(self, current_pose: torch.Tensor, target_action: Optional[torch.Tensor] = None, k: int = 8) -> Optional[torch.Tensor]:
        """
        Get most relevant frames based on flexible retrieval criteria
        Focus: behavioral similarity -> utility -> experience value
        
        Args:
            current_pose: Current position
            target_action: Target action (action to be executed currently)
            k: Number of frames to return
        """
        if len(self.frames) == 0:
            return None
            
        if len(self.frames) <= k:
            return torch.stack(self.frames).to(current_pose.device)
        
        # Calculate retrieval relevance scores (using flexible retrieval criteria)
        retrieval_scores = self.compute_retrieval_score(current_pose, target_action)
        
        # Select top-k
        top_k_indices = torch.topk(retrieval_scores, min(k, len(retrieval_scores))).indices
        used_indices = top_k_indices.tolist()
        
        # Update usage statistics (if frame_counter info available)
        if hasattr(self, 'current_frame_idx'):
            self.update_usage_scores(used_indices, self.current_frame_idx)
        
        relevant_frames = [self.frames[i] for i in top_k_indices]
        return torch.stack(relevant_frames).to(current_pose.device)
    
    def _compute_action_similarity_for_retrieval(self, target_action: torch.Tensor, memory_actions: torch.Tensor) -> torch.Tensor:
        """
        Action similarity calculation specifically designed for retrieval
        Focus: similar actions should find nearest similar execution cases
        """
        target_linear = target_action[:2]
        target_yaw = target_action[2]
        
        memory_linear = memory_actions[:, :2]
        memory_yaw = memory_actions[:, 2]
        
        # 1. Linear motion similarity
        target_linear_norm = torch.norm(target_linear)
        memory_linear_norm = torch.norm(memory_linear, dim=1)
        
        # Motion magnitude similarity
        magnitude_diff = torch.abs(target_linear_norm - memory_linear_norm)
        magnitude_sims = torch.exp(-magnitude_diff / 0.3)  # Stricter magnitude matching
        
        # Motion direction similarity
        direction_sims = torch.ones_like(magnitude_sims)
        if target_linear_norm > 0.1:  # Has obvious linear motion
            target_direction = target_linear / target_linear_norm
            valid_memory = memory_linear_norm > 0.1
            if valid_memory.any():
                memory_directions = memory_linear[valid_memory] / memory_linear_norm[valid_memory].unsqueeze(1)
                dot_products = torch.mm(memory_directions, target_direction.unsqueeze(1)).squeeze()
                direction_sims[valid_memory] = torch.clamp((dot_products + 1) / 2, 0, 1)
        
        # 2. Turning behavior similarity - more precise matching
        yaw_sims = self._compute_precise_rotation_similarity(target_yaw, memory_yaw)
        
        # 3. Comprehensive similarity: for retrieval, we focus more on precise matching
        # Linear motion 50% + direction 25% + turning 25%
        action_similarities = 0.5 * magnitude_sims + 0.25 * direction_sims + 0.25 * yaw_sims
        
        return action_similarities
    
    def _compute_precise_rotation_similarity(self, target_yaw: torch.Tensor, memory_yaw: torch.Tensor) -> torch.Tensor:
        """
        Precise turning similarity calculation for action retrieval
        Focus: same type of turning actions should get high similarity
        """
        config = self.SCORING_CONFIG
        
        target_abs = torch.abs(target_yaw)
        memory_abs = torch.abs(memory_yaw)
        
        # Turning type classification
        def classify_turn(yaw_abs):
            if yaw_abs < 0.1:  # Straight
                return 0
            elif yaw_abs < config['significant_turn_threshold']:  # Fine adjustment
                return 1
            elif yaw_abs < config['sharp_turn_threshold']:  # Turn
                return 2
            else:  # Sharp turn
                return 3
        
        target_class = classify_turn(target_abs)
        memory_classes = torch.tensor([classify_turn(y) for y in memory_abs], device=memory_yaw.device)
        
        # Direction matching
        target_direction = torch.sign(target_yaw)
        memory_directions = torch.sign(memory_yaw)
        direction_match = (target_direction == memory_directions).float()
        
        # Type matching
        target_class_tensor = torch.tensor(target_class, device=memory_yaw.device)
        class_match = (target_class_tensor == memory_classes).float()
        
        # Angle difference
        yaw_diff = torch.abs(target_yaw - memory_yaw)
        angle_similarity = torch.exp(-yaw_diff / 0.2)  # Stricter angle matching
        
        # Comprehensive scoring: exact match > same type same direction > similar angle
        similarities = (0.5 * class_match * direction_match +  # Exact match
                       0.3 * class_match +                    # Same type
                       0.2 * angle_similarity)                # Similar angle
        
        return similarities
    
    def get_memory_stats(self) -> dict:
        """Get memory cache statistics for debugging and monitoring"""
        if len(self.frames) == 0:
            return {"empty": True}
        
        # Analyze dynamic decay system status
        protected_memories = sum(1 for steps in self.unused_steps if steps <= self.SCORING_CONFIG['fixed_memory_time'])
        decaying_memories = sum(1 for steps in self.unused_steps if steps > self.SCORING_CONFIG['fixed_memory_time'])
        avg_unused_steps = sum(self.unused_steps) / len(self.unused_steps) if self.unused_steps else 0
        
        return {
            "total_memories": len(self.frames),
            "max_capacity": self.max_size,
            "average_score": sum(self.scores) / len(self.scores),
            "highest_score": max(self.scores),
            "lowest_score": min(self.scores),
            "total_usage": sum(self.usage_counts),
            "most_used_count": max(self.usage_counts) if self.usage_counts else 0,
            # Dynamic decay system statistics
            "protected_memories": protected_memories,      # Number of memories enjoying protection period
            "decaying_memories": decaying_memories,        # Number of memories currently decaying
            "avg_unused_steps": avg_unused_steps,          # Average consecutive unused steps
            "max_unused_steps": max(self.unused_steps) if self.unused_steps else 0,
            "fixed_memory_time": self.SCORING_CONFIG['fixed_memory_time'],
            "accelerated_decay_rate": self.SCORING_CONFIG['accelerated_decay_rate'],
            "scoring_config": self.SCORING_CONFIG
        }
    
    def get_adaptive_scoring_stats(self, current_pose: torch.Tensor, target_action: Optional[torch.Tensor] = None) -> dict:
        """
        Get adaptive scoring system statistics
        """
        if target_action is None:
            return {"adaptive_scoring": "disabled", "reason": "no_target_action"}
        
        current_situation = {
            'action': target_action,
            'pose': current_pose
        }
        
        complexity = self.adaptive_scorer.compute_situation_complexity(current_situation)
        adaptive_weights = self.adaptive_scorer.compute_adaptive_weights(current_situation)
        
        # Compare default weights and adaptive weights
        default_weights = {
            'action': self.SCORING_CONFIG['retrieval_action_weight'],
            'memory': self.SCORING_CONFIG['retrieval_memory_weight'],
            'spatial': self.SCORING_CONFIG['retrieval_spatial_weight'],
            'usage': self.SCORING_CONFIG['retrieval_usage_weight']
        }
        
        return {
            "complexity_score": complexity,
            "use_adaptive": complexity > 0.3,
            "default_weights": default_weights,
            "adaptive_weights": {
                'action': adaptive_weights['retrieval_action_weight'],
                'memory': adaptive_weights['retrieval_memory_weight'],
                'spatial': adaptive_weights['retrieval_spatial_weight'],
                'usage': adaptive_weights['retrieval_usage_weight']
            },
            "weight_differences": {
                'action': adaptive_weights['retrieval_action_weight'] - default_weights['action'],
                'memory': adaptive_weights['retrieval_memory_weight'] - default_weights['memory'],
                'spatial': adaptive_weights['retrieval_spatial_weight'] - default_weights['spatial'],
                'usage': adaptive_weights['retrieval_usage_weight'] - default_weights['usage']
            }
        }
    
    def reset_memory(self):
        """Reset memory cache"""
        self.frames.clear()
        self.poses.clear()
        self.actions.clear()
        self.frame_indices.clear()
        self.scores.clear()
        self.usage_counts.clear()
        self.last_used.clear()
        self.unused_steps.clear()  # Reset consecutive unused steps counter
    

class SelectiveMemoryAttention(nn.Module):
    """
    Selective Memory Attention Module - LT-NWM Core Innovation
    
    Implements dual-filter mechanism:
    - Retrieval from memory buffer 
    - Relevance-based attention threshold (> 10) to filter noise
    
    Multi-head attention processing of 8 key historical frames
    Only activated in relevant scenes (score > 30) to avoid noise
    """
    def __init__(self, hidden_size: int, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Memory query/key/value projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism for optional activation
        self.activation_gate = nn.Parameter(torch.zeros(1))
        self.relevance_threshold = 10.0  # Dual-filter mechanism: relevance-based attention threshold (> 10)
        
    def compute_relevance(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores between query and memory
        Part of dual-filter mechanism for noise filtering
        """
        # Enhanced relevance calculation: cosine similarity scaled to meaningful range
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        memory_norm = torch.nn.functional.normalize(memory, dim=-1)
        
        # Cosine similarity returns [-1, 1], scale to [0, 100] for threshold compatibility
        cosine_sim = torch.sum(query_norm.unsqueeze(1) * memory_norm, dim=-1)
        relevance_score = (cosine_sim + 1.0) * 50.0  # Scale from [-1,1] to [0,100]
        
        return relevance_score
    
    def forward(self, x: torch.Tensor, memory_frames: Optional[torch.Tensor] = None, 
                activate_memory: bool = True) -> torch.Tensor:
        """
        Forward pass with optional memory activation
        
        Args:
            x: Current frame features [B, N, D]
            memory_frames: Memory buffer frames [B, M, N, D] 
            activate_memory: Whether to activate memory mechanism
        """
        if not activate_memory or memory_frames is None:
            return x * 0  # Return zero if memory not activated
        
        B, N, D = x.shape
        _, M, _, _ = memory_frames.shape
        
        # Reshape memory for attention
        memory_flat = memory_frames.view(B, M * N, D)
        
        # Dual-filter mechanism: compute relevance and filter noise
        relevance = self.compute_relevance(x.mean(dim=1, keepdim=True), memory_flat.mean(dim=1, keepdim=True))
        
        # Relevance-based attention threshold (> 10) to filter noise
        if relevance.max() < self.relevance_threshold:
            return x * 0  # Skip memory if not relevant enough (dual-filter mechanism)
        
        # Multi-head attention computation
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory_flat).view(B, M * N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory_flat).view(B, M * N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(out)
        
        # Apply activation gate (learnable parameter)
        return out * torch.sigmoid(self.activation_gate)


class HybridCDiTBlock(nn.Module):
    """
    Hybrid Three-Layer Attention Architecture - LT-NWM Core Innovation
    
    The standard CDiT attention is augmented with selective memory attention:
    1. Self-Attention (standard CDiT layer)
    2. Cross-Attention (4 frames - CDiT core strength) 
    3. Selective Memory Attention (8 key historical frames - WorldMem enhancement)
    
    Combines CDiT's precise conditional control with long-term memory consistency
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, 
                 enable_memory: bool = True, **block_kwargs):
        super().__init__()
        self.enable_memory = enable_memory
        
        # CDiT core components (unchanged)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                         add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        
        # Memory components (new)
        if enable_memory:
            self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.memory_attn = SelectiveMemoryAttention(hidden_size, num_heads)
        
        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = nn.GELU
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # AdaLN modulation (extended for memory)
        num_modulations = 14 if enable_memory else 11
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_modulations * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, x_cond: torch.Tensor, 
                memory_frames: Optional[torch.Tensor] = None, 
                memory_activation_score: float = 0.0):
        """
        Forward pass with optional memory integration
        
        Args:
            x: Input features [B, N, D]
            c: Conditioning signal [B, D] 
            x_cond: Context frames [B, context_size, N, D]
            memory_frames: Memory buffer [B, M, N, D]
            memory_activation_score: Score determining memory activation strength
        """
        if self.enable_memory:
            modulations = self.adaLN_modulation(c).chunk(14, dim=1)
            (shift_msa, scale_msa, gate_msa, 
             shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x,
             shift_mem, scale_mem, gate_mem,
             shift_mlp, scale_mlp, gate_mlp) = modulations
        else:
            modulations = self.adaLN_modulation(c).chunk(11, dim=1)
            (shift_msa, scale_msa, gate_msa, 
             shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x,
             shift_mlp, scale_mlp, gate_mlp) = modulations
        
        # 1. Self-attention (CDiT standard)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # 2. Cross-attention with immediate context (CDiT core strength)
        x_cond_norm = modulate(self.norm_cond(x_cond.flatten(1, 2)), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), 
            key=x_cond_norm, 
            value=x_cond_norm, 
            need_weights=False
        )[0]
        
        # 3. Selective memory attention (WorldMem enhancement)
        if self.enable_memory and memory_frames is not None:
            # Adaptive memory activation based on relevance score - only activated in relevant scenes (score > 30)
            activate_memory = memory_activation_score > 30.0  # Adaptive activation threshold for relevant scenes
            
            memory_output = self.memory_attn(
                modulate(self.memory_norm(x), shift_mem, scale_mem),
                memory_frames=memory_frames,
                activate_memory=activate_memory
            )
            x = x + gate_mem.unsqueeze(1) * memory_output
        
        # 4. MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        
        return x


class HybridCDiT(nn.Module):
    """
    Hybrid CDiT model combining precise conditional control with selective long-term memory
    """
    def __init__(
        self,
        input_size: int = 32,
        context_size: int = 4,  # Will be 4 for XL, 3 for L/B
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        memory_enabled: bool = True,
        memory_layers: Optional[List[int]] = None,
        memory_buffer_size: int = 50
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.memory_enabled = memory_enabled
        
        # Default memory layers (activate in later layers)
        if memory_layers is None:
            memory_layers = list(range(depth // 2, depth))  # Second half of layers
        self.memory_layers = set(memory_layers)
        
        # Core CDiT components
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)
        
        # Positional embeddings
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size), 
            requires_grad=True
        )
        
        # Transformer blocks (hybrid)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            enable_memory_layer = memory_enabled and (i in self.memory_layers)
            self.blocks.append(
                HybridCDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                              enable_memory=enable_memory_layer)
            )
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Memory management
        self.memory_buffer = MemoryBuffer(max_size=memory_buffer_size) if memory_enabled else None
        self.frame_counter = 0
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights (same as CDiT)"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize embedders
        for embedder in [self.y_embedder.x_emb, self.y_embedder.y_emb, self.y_embedder.angle_emb, 
                        self.t_embedder, self.time_embedder]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def update_memory(self, frame_latent: torch.Tensor, pose: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Intelligently update memory cache, deciding storage based on scoring system"""
        if self.memory_buffer is not None:
            # Use intelligent storage mechanism
            stored = self.memory_buffer.add_frame(frame_latent, pose, action, self.frame_counter)
            # Update current frame index for score decay calculation
            self.memory_buffer.current_frame_idx = self.frame_counter
            self.frame_counter += 1
            return stored
        return False
    
    def get_memory_stats(self):
        """Get memory system statistics"""
        if self.memory_buffer is not None:
            return self.memory_buffer.get_memory_stats()
        return {"memory_disabled": True}
    
    def compute_memory_activation_score(self, current_pose: torch.Tensor, 
                                       action_magnitude: float) -> float:
        """
        Compute a score that determines when to activate memory
        Higher scores indicate more need for memory consultation
        """
        # Activate memory more when:
        # 1. Large camera movements (might revisit areas)
        # 2. Low action magnitude (might be looking around)
        # 3. When memory buffer has sufficient content
        
        movement_score = min(action_magnitude / 5.0, 1.0)  # Normalize action magnitude
        buffer_score = len(self.memory_buffer.frames) / self.memory_buffer.max_size if self.memory_buffer else 0
        
        # Combination heuristic (can be learned)
        activation_score = 0.3 * (1 - movement_score) + 0.7 * buffer_score
        return activation_score
    
    def unpatchify(self, x):
        """Unpatchify as in original CDiT"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, int(h * p), int(w * p)))
        return imgs
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
                x_cond: torch.Tensor, rel_t: torch.Tensor, 
                current_pose: Optional[torch.Tensor] = None,
                update_memory: bool = True):
        """
        Forward pass with optional memory integration
        
        Args:
            x: Input tensor [N, C, H, W]
            t: Timestep [N]
            y: Action conditions [N, 3]  
            x_cond: Context frames [N, context_size, C, H, W]
            rel_t: Relative time [N]
            current_pose: Current camera pose [N, 4] (x,y,z,yaw) - no pitch in dataset
            update_memory: Whether to update memory buffer
        """
        # Embed inputs (same as CDiT)
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]
        
        # Conditioning
        t_emb = self.t_embedder(t[..., None])
        y_emb = self.y_embedder(y)
        time_emb = self.time_embedder(rel_t[..., None])
        c = t_emb + time_emb + y_emb
        
        # Memory preparation - only use during inference, skip during training
        memory_frames = None
        memory_activation_score = 0.0
        
        # Only use memory buffer during inference phase (not training mode) on GPU cluster
        if self.memory_enabled and current_pose is not None and not self.training and self.memory_buffer is not None:
            # Get relevant memory frames based on intended action (behavioral similarity)
            target_action = y[0] if y is not None else None  # Current target action
            memory_frames = self.memory_buffer.get_relevant_frames(
                current_pose[0], target_action=target_action, k=8
            )
            if memory_frames is not None:
                # memory_frames shape: [k, C, H, W], need to expand to [batch_size, k, C, H, W]
                if memory_frames.dim() == 4:  # [k, C, H, W]
                    memory_frames = memory_frames.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
                elif memory_frames.dim() == 3:  # [k, H, W] single channel case
                    memory_frames = memory_frames.unsqueeze(0).unsqueeze(2).expand(x.shape[0], -1, x.shape[1], -1, -1)
            
            # Compute memory activation score
            action_magnitude = torch.norm(y[0]).item()
            memory_activation_score = self.compute_memory_activation_score(current_pose[0], action_magnitude)
        
        # Transformer blocks with selective memory
        for i, block in enumerate(self.blocks):
            if i in self.memory_layers:
                # Use memory for inference in specified layers
                x = block(x, c, x_cond, memory_frames, memory_activation_score)
            else:
                # Standard CDiT processing (no memory usage for inference)
                x = block(x, c, x_cond)
        
        # Final processing
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        # Update memory during inference for all layers (storage happens regardless of layer)
        # Key design: 
        # - Layers 0-15 (non-memory): Store to memory but don't use memory for inference
        # - Later layers (memory): Both store to memory and use memory for inference
        # This ensures continuous memory building even when only using CDiT
        if update_memory and self.memory_enabled and current_pose is not None and not self.training:
            current_action = y[0] if y is not None else None  # Store the action that led to this frame
            self.update_memory(x.detach(), current_pose[0], current_action)
        
        return x


# Model configurations
def HybridCDiT_XL_2(**kwargs):
    return HybridCDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, 
                     context_size=4, **kwargs)

def HybridCDiT_L_2(**kwargs):
    return HybridCDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, 
                     context_size=3, **kwargs)

def HybridCDiT_B_2(**kwargs):
    return HybridCDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, 
                     context_size=3, **kwargs)

def HybridCDiT_S_2(**kwargs):
    return HybridCDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, 
                     context_size=3, **kwargs)

HybridCDiT_models = {
    'HybridCDiT-XL/2': HybridCDiT_XL_2,
    'HybridCDiT-L/2': HybridCDiT_L_2, 
    'HybridCDiT-B/2': HybridCDiT_B_2,
    'HybridCDiT-S/2': HybridCDiT_S_2
}

if __name__ == "__main__":
    # Test the hybrid model
    model = HybridCDiT_L_2(memory_enabled=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, 3)
    x_cond = torch.randn(batch_size, 3, 4, 32, 32)  # context_size=3 for L model
    rel_t = torch.randn(batch_size)
    current_pose = torch.randn(batch_size, 4)  # [x, y, z, yaw] - no pitch
    
    with torch.no_grad():
        output = model(x, t, y, x_cond, rel_t, current_pose)
        print(f"Output shape: {output.shape}")
