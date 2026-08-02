# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
from retrieval_context import load_revisit_manifest, select_context_indices

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if isinstance(predefined_index, (list, tuple)):
            self.index_to_data = list(predefined_index)
            return
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )
            
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index
  
    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos    

class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)


    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)

class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)
  
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1])) # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class RevisitEvalDataset(EvalDataset):
    """Manifest-driven evaluation dataset with fixed four-slot context policies."""

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        manifest_path: str,
        context_policy: str,
        selection_seed: int,
        position_scale: float = 0.75,
        heading_scale_radians: float = np.deg2rad(20.0),
        max_queries: int = 0,
        traj_names: str = "traj_names.txt",
        normalize: bool = True,
        goals_per_obs: int = 1,
    ):
        if context_size != 4:
            raise ValueError(
                f"Phase A context replacement requires context_size=4, got {context_size}"
            )
        manifest = load_revisit_manifest(manifest_path)
        names_path = os.path.join(data_split_folder, traj_names)
        with open(names_path, "r", encoding="utf-8") as stream:
            allowed_trajectories = {line.strip() for line in stream if line.strip()}

        records = []
        selections = []
        diagnostics = []
        index_rows = []
        trajectory_cache = {}

        for record in manifest:
            trajectory = record["trajectory"]
            if trajectory not in allowed_trajectories:
                continue
            if trajectory not in trajectory_cache:
                trajectory_path = os.path.join(data_folder, trajectory, "traj_data.pkl")
                with open(trajectory_path, "rb") as stream:
                    metadata = pickle.load(stream)
                if not isinstance(metadata, dict) or not {"position", "yaw"} <= metadata.keys():
                    raise ValueError(
                        f"{trajectory_path}: expected position and yaw trajectory metadata"
                    )
                trajectory_cache[trajectory] = (
                    np.asarray(metadata["position"], dtype=np.float64),
                    np.asarray(metadata["yaw"], dtype=np.float64).squeeze(),
                )

            positions, yaws = trajectory_cache[trajectory]
            query_index = int(record["query_index"])
            if query_index + len_traj_pred >= len(positions):
                raise ValueError(
                    f"{trajectory}:{query_index} has fewer than {len_traj_pred} future frames; "
                    "rebuild the manifest or reduce the requested evaluation horizon"
                )
            selection = select_context_indices(
                policy=context_policy,
                trajectory=trajectory,
                query_index=query_index,
                context_size=context_size,
                seed=selection_seed,
                positive_indices=record.get("positive_indices", ()),
                positions=positions,
                yaws=yaws,
                min_temporal_gap=int(record.get("min_temporal_gap", context_size)),
                position_scale=position_scale,
                heading_scale_radians=heading_scale_radians,
            )
            diagnostic = {
                "manifest_index": len(records),
                "trajectory": trajectory,
                "query_index": query_index,
                "ground_truth_future_index": query_index + len_traj_pred,
                "ground_truth_indices": list(
                    range(query_index + 1, query_index + len_traj_pred + 1)
                ),
                "label_source": record.get("label_source", "unknown"),
                "selection_seed": selection_seed,
                **selection.to_dict(),
                "selected_source_path": get_data_path(
                    data_folder, trajectory, selection.selected_source_index
                ),
                "ground_truth_future_path": get_data_path(
                    data_folder, trajectory, query_index + len_traj_pred
                ),
            }
            records.append(record)
            selections.append(selection)
            diagnostics.append(diagnostic)
            index_rows.append((trajectory, query_index, 0, len_traj_pred))
            if max_queries > 0 and len(records) >= max_queries:
                break

        if not records:
            raise ValueError(
                f"{manifest_path}: no records belong to trajectories in {names_path}"
            )

        self.manifest_path = str(manifest_path)
        self.context_policy = context_policy
        self.selection_seed = selection_seed
        self._manifest_records = records
        self._context_selections = selections
        self._diagnostics = diagnostics

        super().__init__(
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name=dataset_name,
            image_size=image_size,
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            len_traj_pred=len_traj_pred,
            traj_stride=traj_stride,
            context_size=context_size,
            transform=transform,
            traj_names=traj_names,
            normalize=normalize,
            predefined_index=index_rows,
            goals_per_obs=goals_per_obs,
        )

    def get_diagnostic(self, index: int):
        return dict(self._diagnostics[index])

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            trajectory, query_index, _, _ = self.index_to_data[i]
            selection = self._context_selections[i]
            pred_times = list(
                range(query_index + 1, query_index + self.len_traj_pred + 1)
            )
            obs_image = torch.stack(
                [
                    self.transform(
                        Image.open(get_data_path(self.data_folder, trajectory, frame_index))
                    )
                    for frame_index in selection.context_indices
                ]
            )
            pred_image = torch.stack(
                [
                    self.transform(
                        Image.open(get_data_path(self.data_folder, trajectory, frame_index))
                    )
                    for frame_index in pred_times
                ]
            )

            trajectory_data = self._get_trajectory(trajectory)
            actions, _ = self._compute_actions(
                trajectory_data, query_index, np.array([query_index + 1])
            )
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32),
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as error:
            print(f"Exception in revisit dataset {self.dataset_name}", error)
            raise
        
class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

   
    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))           
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
