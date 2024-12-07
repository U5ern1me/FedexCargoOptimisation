import copy

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO
import logging

# typing
from typing import List, Any, Tuple, Dict


class Package:
    """
    Package overwrite for the DRL solver
    """

    def __init__(
        self,
        id: str,
        length: int,
        width: int,
        height: int,
        weight: int,
        Priority: str,
        delay_cost: int,
    ):
        """
        Args:
            id: Package ID
            length: Package length
            width: Package width
            height: Package height
            weight: Package weight
        """
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        if Priority == "Priority":
            self.priority = 1
        else:
            self.priority = 0
        if self.priority == 0:
            self.delay_cost = int(delay_cost)
        else:
            self.delay_cost = 100000
        self.current_ULD = None
        self.current_postion = None
        self.orientation = None

    def update_ULD(self, uld_id: str, current_position: List[int], orientaion: int):
        """
        Update the ULD of the package

        Args:
            uld_id: ULD ID
            current_position: Current position of the package
            orientaion: Orientation of the package
        """
        self.current_ULD = uld_id
        self.current_postion = current_position
        self.orientation = orientaion

    def orient(self, orientation: int) -> Tuple[int, int, int]:
        """
        return the orientated package dimensions based on the orientation

        Args:
            orientation: Orientation of the package

        Returns:
            Length, width, and height of the orientated package
        """
        length = self.length
        width = self.width
        height = self.height
        if orientation == 1:
            pass
        elif orientation == 2:
            length, height, width = height, width, length
        elif orientation == 3:
            length, height, width = width, length, height
        elif orientation == 4:
            length, width, height = width, length, height
        elif orientation == 5:
            length, width, height = length, height, width
        elif orientation == 6:
            length, width, height = height, width, length
        return length, width, height

    def reset(self):
        """
        Reset the package to its initial state
        """
        self.current_ULD = None
        self.current_postion = None
        self.orientation = None


class Package_DRL:
    """
    Package list for the DRL solver
    """

    def __init__(self, packages: List[Package]):
        """
        Args:
            packages: List of packages
        """
        self.package_ids = []
        self.priority_count = 0
        self.packages = []
        for package in packages:
            self.package_ids.append(package.id)
            self.packages.append(
                Package(
                    package.id,
                    package.length,
                    package.width,
                    package.height,
                    package.weight,
                    package.priority,
                    package.delay_cost,
                )
            )
            if self.packages[-1].priority == 1:
                self.priority_count += 1

    def reset(self):
        """
        Reset the packages to their initial state
        """
        for package in self.packages:
            package.reset()


class ULD:
    """
    ULD overwrite for the DRL solver
    """

    def __init__(
        self, id: str, length: int, width: int, height: int, max_weight_limit: int
    ):
        """
        Args:
            id: ULD ID
            length: ULD length
            width: ULD width
            height: ULD height
            max_weight_limit: Maximum weight limit of the ULD
        """
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.max_weight_limit = max_weight_limit
        self.max_volume = length * width * height
        self.remaining_weight = max_weight_limit
        self.remaining_volume = self.max_volume
        self.package_assignments = []

    def update_package_assignment(self, package_: Package, orientation: int) -> None:
        """
        Update the package assignment of the ULD

        Args:
            package_: Package
            orientation: Orientation of the package
        """
        length, width, height = package_.orient(orientation)
        self.package_assignments.append(
            {
                "Pack_map": package_,
                "x1": package_.current_postion[0],
                "y1": package_.current_postion[1],
                "z1": package_.current_postion[2],
                "x2": package_.current_postion[0] + length,
                "y2": package_.current_postion[1] + width,
                "z2": package_.current_postion[2] + height,
            }
        )

        self.remaining_weight -= package_.weight
        self.remaining_volume -= length * width * height

    def reset(self):
        """
        Reset the ULD to its initial state
        """
        self.remaining_weight = self.max_weight_limit
        self.remaining_volume = self.max_volume
        self.package_assignments = []


class ULD_DRL:
    """
    ULD list for the DRL solver
    """

    def __init__(self, ulds: List[ULD]):
        """
        Args:
            ulds: List of ULDs
        """
        self.ulds = []
        for uld in ulds:
            self.ulds.append(
                ULD(uld.id, uld.length, uld.width, uld.height, uld.weight_limit)
            )

    def reset(self):
        """
        Reset the ULD to its initial state
        """
        for uld in self.ulds:
            uld.reset()


class Bin_Packing_Env(gym.Env):
    """
    Bin Packing Environment for the DRL solver
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        ulds_: ULD_DRL,
        packages_: Package_DRL,
        ulds_solver: List[ULD],
        packages_solver: List[Package],
    ):
        """
        Args:
            ulds_: ULD_DRL
            packages_: Package_DRL
            ulds_solver: List of ULDs
            packages_solver: List of Packages
        """
        super(Bin_Packing_Env, self).__init__()
        self.seed_value = None
        self.step_count = 0
        self.max = 0
        self.solver_uld = ulds_solver
        self.solver_packages = packages_solver
        self.uld_data = copy.deepcopy(ulds_)
        self.packages_ = copy.deepcopy(packages_)
        self.packages_placed = 0
        self.packages_left = len(packages_.package_ids)
        self.packages_placed_list = []
        self.ulds_used_for_Priority = []
        self.placed = np.zeros(shape=(len(self.packages_.package_ids),))

        ## Calculate the maximum dimensions for the DRL solver
        length, width, height, self.max_weight_limit = calculate_max_dim(
            self.uld_data.ulds
        )

        ## Set the maximum dimensions for the DRL solver
        self.max_l = length
        self.max_w = width
        self.max_h = height

        ## Calculate the diagonal for the DRL solver for the reward function
        self.diagonal = (length**2 + width**2) ** 0.5

        ## Set the action space for the DRL solver
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(5,))

        ## Initialize the rewards list for the DRL solver
        self.rewards = []

        ## Calculate the total priority available for the DRL solver
        self.total_priority_available = 0
        for package in self.packages_.packages:
            if package.priority:
                self.total_priority_available += 1

        ## Calculate the height map for the DRL solver
        self.height_map = Height_map(self.uld_data)

        ## Calculate the observation space for the DRL solver
        Height_dim = 0
        for height_map in self.height_map:
            Height_dim += height_map.shape[0] * height_map.shape[1]

        ## Set the observation space for the DRL solver
        self.observation_space = spaces.Box(
            low=0,
            high=10000,
            shape=(
                Height_dim
                + (len(self.packages_.package_ids) * 6)
                + len(self.uld_data.ulds)
                + self.placed.shape[0],
            ),
            dtype=np.float32,
        )

    def mask_action(self, action: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Mask the action based on the termination and info

        Args:
            action: Action

        Returns:
            Termination (bool), info (Dict[str, Any])
        """
        termination = False
        info = {}

        ## Extract the action values
        package_index = action[1]
        uld_index = action[0]
        orientation = action[2]
        x = action[3]
        y = action[4]

        ## Check if the package is already placed
        if package_index in self.packages_placed_list:
            termination = True
            info = {
                "Terminated": True,
                "Termination_Reason": "Package already placed",
                "message": f"Termination : package already placed whilst placing {self.packages_.package_ids[package_index]} in uld : {self.uld_data.ulds[uld_index].id}",
                "data": {},
            }
            return termination, info

        ## Check if the package weight is greater than the remaining weight of the ULD
        if (
            self.packages_.packages[package_index].weight
            > self.uld_data.ulds[uld_index].remaining_weight
        ):
            termination = True
            info = {
                "Terminated": True,
                "Termination_Reason": "Weight limit exceeded",
                "message": f"Termination : Weight limit exceeded whilst placing {self.packages_.package_ids[package_index]} in uld : {self.uld_data.ulds[uld_index].id}",
                "data": {},
            }
            return termination, info

        ## Calculate the orientated package dimensions
        length, width, height = self.packages_.packages[package_index].orient(
            orientation
        )

        ## Check if the package is contractable
        height_map = self.height_map[uld_index]
        if (
            x + length - 1 >= height_map.shape[0]
            or y + width - 1 >= height_map.shape[1]
        ):
            termination = True
            info = {
                "Terminated": True,
                "Termination_Reason": "Contractability issues",
                "message": f"Termination : Contractability issues exceeded whilst placing {self.packages_.package_ids[package_index]} in uld : {self.uld_data.ulds[uld_index].id}",
                "data": {},
            }
            return termination, info

        ## Calculate the height at the given coordinates
        height_at_xy = height_map[x, y].item()

        ## Check if the height of the package is greater than the height of the ULD
        if height_at_xy + height > self.uld_data.ulds[uld_index].height:
            termination = True
            info = {
                "Terminated": True,
                "Termination_Reason": "Height exceeded",
                "message": f"Termination : Height exceeded whilst placing {self.packages_.package_ids[package_index]} in uld : {self.uld_data.ulds[uld_index].id}",
                "data": {},
            }
            return termination, info

        ## Check if the package overlaps with any other package
        for i in range(x, x + length):
            for j in range(y, y + width):
                if height_map[i][j] > height_at_xy:
                    termination = True
                    info = {
                        "Terminated": True,
                        "Termination_Reason": "Overlap issues",
                        "message": f"Termination : Overlap issues exceeded whilst placing {self.packages_.package_ids[package_index]} in uld : {self.uld_data.ulds[uld_index].id}",
                        "data": {},
                    }
                    return termination, info
        return termination, info

    def calculate_reward(
        self, termination: bool, action: np.ndarray, reason: str
    ) -> float:
        """
        Calculate the reward for the DRL solver

        Args:
            termination: Termination (bool)
            action: Action (np.ndarray)
            reason: Reason (str)

        Returns:
            Reward
        """
        reward = 0

        ## Extract the action values
        uld_index = action[0]
        package_index = action[1]
        orientation = action[2]
        x = action[3]
        y = action[4]

        ## Calculate the reward based on the termination and reason
        if termination:
            if reason == "Package already placed":
                reward = 0
            elif reason == "Overlap issues":
                reward = 0
            elif reason == "Weight limit exceeded":
                reward = 0
            elif reason == "Contractability issues":
                reward = 0
            elif reason == "Height exceeded":
                reward = 0
            else:
                reward = 0
        else:
            ## Calculate the priority factor discounting
            if self.priority_placed < self.total_priority_available:
                priority_factor_discounting = 0.5
            else:
                priority_factor_discounting = 1

            ## Calculate the height map
            height_map = self.height_map[uld_index]
            after_update_height = height_map[action[3]][action[4]].item()

            ## Calculate the reward
            reward += package_index * 10
            reward += (x + y) * 5
            reward += 10 * (
                (
                    self.uld_data.ulds[uld_index].max_volume
                    - self.uld_data.ulds[uld_index].remaining_volume
                )
                / self.uld_data.ulds[uld_index].max_volume
            )
            reward += (
                self.uld_data.ulds[uld_index].height - after_update_height
            ) / self.uld_data.ulds[uld_index].height
            reward += 10
            if self.packages_.packages[package_index].priority != 1:
                reward *= priority_factor_discounting

        ## Update the maximum packages placed
        if self.packages_placed > self.max:
            self.max = self.packages_placed

        return reward

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes an action and updates the environment.

        Args:
            action: Action (np.ndarray)

        Returns:
            Observation (np.ndarray), Reward (float), Terminated (bool), Truncated (bool), Info (Dict[str, Any])
        """
        ## DENORMALIZING THE ACTION
        action[0] = abs(action[0]) * (len(self.uld_data.ulds) - 1)
        action[1] = abs(action[1]) * (len(self.packages_.package_ids) - 1)
        action[2] = abs(action[2]) * 5 + 1
        action[3] = abs(action[3]) * self.max_l
        action[4] = abs(action[4]) * self.max_w

        ## Convert the action to integer
        action = action.astype(np.int32)

        ## Update the step count
        self.step_count += 1

        ## Initialize the terminated, info, reward, observation and truncated
        self.terminated = False
        self.info = {}
        self.reward = 0
        self.observation = []
        self.truncated = False
        self.terminated, self.info = self.mask_action(action)

        ## If the package is terminated
        if self.terminated:
            self.observation = self.get_observation()
            self.reward = self.calculate_reward(
                self.terminated, action, self.info["Termination_Reason"]
            )
            self.rewards.append(self.reward)
            self.reward = float(self.reward)
            return self.observation, self.reward, False, False, self.info
        else:
            ## Update the environment
            self.update_environment(action)

            ## Check if all the packages are placed
            if self.packages_left == 0:
                self.terminated = True
                self.info = {
                    "Terminated": True,
                    "Termination_Reason": "Successfully placed all the packages",
                    "message": "Successfully placed all the packages",
                }
            else:
                self.terminated = False
                self.info = {
                    "Terminated": False,
                    "Termination_Reason": "Successfully placed a package",
                    "message": "Successfully placed a package",
                }
            ## Calculate the reward
            self.reward = self.calculate_reward(
                self.terminated, action, self.info["Termination_Reason"]
            )
            self.reward = float(self.reward)
            self.rewards.append(self.reward)
            self.observation = self.get_observation()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def update_environment(self, action: np.ndarray):
        """
        Update the environment based on the action

        Args:
            action: Action (np.ndarray)
        """
        ## Update the package ULD
        self.packages_.packages[action[1]].update_ULD(
            self.uld_data.ulds[action[0]].id,
            [
                action[3],
                action[4],
                self.height_map[action[0]][action[3]][action[4]].item(),
            ],
            action[2],
        )

        ## Update the packages placed list
        self.packages_placed_list.append([action[1]])

        ## Update the ULD package assignment
        self.uld_data.ulds[action[0]].update_package_assignment(
            self.packages_.packages[action[1]], action[2]
        )

        ## Update the solver ULD package assignment
        self.solver_uld[action[0]].packages.append(
            self.packages_.packages[action[1]].id
        )

        ## Update the solver package ULD assignment
        self.solver_packages[action[1]].uld_id = self.uld_data.ulds[action[0]].id
        self.solver_packages[action[1]].point1 = [
            action[3],
            action[4],
            self.height_map[action[0]][action[3]][action[4]].item(),
        ]

        length, width, height = self.packages_.packages[action[1]].orient(action[2])
        ## Update the solver package point 2    
        self.solver_packages[action[1]].point2 = [
            action[3] + length,
            action[4] + width,
            self.height_map[action[0]][action[3]][action[4]].item()
            + height,
        ]

        ## Update the packages left and placed
        self.packages_left -= 1
        self.packages_placed += 1
        self.placed[action[1]] = 1

        ## Update the priority placed and ULD used for priority
        if self.packages_.packages[action[1]].priority:
            if self.uld_data.ulds[action[0]].id not in self.ulds_used_for_Priority:
                self.ulds_used_for_Priority.append(self.uld_data.ulds[action[0]].id)

        ## Calculate the orientated package dimensions
        package_length, package_width, package_height = self.packages_.packages[
            action[1]
        ].orient(action[2])

        ## Calculate the height map
        height_map = self.height_map[action[0]]
        height_to_set = height_map[action[3]][action[4]].item() + package_height

        ## Update the height map
        for i in range(action[3], action[3] + package_length):
            for j in range(action[4], action[4] + package_width):
                height_map[i][j] = height_to_set

    def get_observation(self) -> np.ndarray:
        """
        Get the observation for the DRL solver

        Returns:
            Observation (np.ndarray)
        """
        ## Initialize the observation
        self.observation = []
        uld = 0

        ## Flatten the height map
        for height_map in self.height_map:
            flat_height_map = torch.flatten(height_map).tolist()
            flat_height_map = normalize_list(flat_height_map, self.max_h)
            self.observation.extend(flat_height_map)
            self.observation.append(
                float(self.uld_data.ulds[uld].remaining_weight)
                / self.uld_data.ulds[uld].max_weight_limit
            )
            uld += 1

        ## Flatten the packages
        for package in self.packages_.packages:
            self.observation.extend(
                [
                    float(package.length) / self.max_l,
                    float(package.width) / self.max_w,
                    float(package.height) / self.max_h,
                    float(package.weight) / self.max_weight_limit,
                    float(package.priority),
                    float(package.delay_cost) / 10000,
                ]
            )

        ## Flatten the placed packages
        self.observation.extend(self.placed.tolist())

        ## Convert the observation to numpy array
        self.observation = np.array(self.observation, dtype=np.float32)

        return self.observation

    def seed(self, seed=None):
        """
        Seed the environment for the DRL solver
        """
        self.seed_value = seed
        if self.seed_value == None:
            self.seed_value = np.random.seed(self.seed_value)

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for the DRL solver

        Returns:
            Observation (np.ndarray), Info (Dict[str, Any])
        """
        ## Initialize the terminated and info
        self.terminated = False
        info = {"message": "Resetting Env....", "Data": {}}

        ## Reset the ULD and packages
        self.uld_data.reset()
        self.packages_.reset()

        ## Calculate the height map
        self.height_map = Height_map(self.uld_data)

        ## Initialize the packages placed and left
        self.packages_placed = 0
        self.packages_left = len(self.packages_.package_ids)
        self.packages_placed_list = []
        self.ulds_used_for_Priority = []
        self.placed = np.zeros(shape=(len(self.packages_.package_ids),))
        self.priority_placed = 0

        ## Calculate the observation
        self.observation = self.get_observation()

        return self.observation, info

    def close(self):
        """
        Close the environment for the DRL solver
        """
        ## Delete the height map, ULD, packages, observation, rewards, placed, ULD used for priority and packages placed list
        del self.height_map
        del self.uld_data
        del self.packages_
        del self.observation
        del self.rewards
        del self.placed
        del self.ulds_used_for_Priority
        del self.packages_placed_list
        del self.packages_left
        del self.packages_placed
        del self.priority_placed


def Height_map(ulds_: ULD_DRL) -> List[torch.Tensor]:
    """
    Create the height map for the DRL solver

    Args:
        ulds_: ULD_DRL

    Returns:
        Height map (List[torch.Tensor])
    """
    tensor_list = []
    for uld in ulds_.ulds:
        tensor_list.append(torch.zeros(size=(uld.length, uld.width)))
    return tensor_list


def calculate_max_dim(ulds_: List[ULD]) -> Tuple[int, int, int, int]:
    """
    Calculate the maximum dimensions for the DRL solver

    Args:
        ulds_: List of ULDs

    Returns:
        Length, width, height, and max weight limit
    """
    length = 0
    width = 0
    height = 0
    max_weight_limit = 0
    for uld in ulds_:
        if uld.length > length:
            length = uld.length
        if uld.width > width:
            width = uld.width
        if uld.height > height:
            height = uld.height
        if uld.max_weight_limit > max_weight_limit:
            max_weight_limit = uld.max_weight_limit
    return length, width, height, max_weight_limit


def normalize_list(list: List[float], max_value: float) -> List[float]:
    """
    Normalize the list for the DRL solver

    Args:
        list: List of values
        max_value: Maximum value

    Returns:
        Normalized list
    """
    return [x / max_value for x in list]


class DRL_Model(nn.Module):
    def __init__(self):
        super(DRL_Model, self).__init__()

    def train(
        self,
        ULD_data: List[ULD],
        Package_data: List[Package],
        model_params: Dict[str, Any],
    ) -> str:
        """
        Solve using DRL model

        Args:
            ULD_data: List of ULDs
            Package_data: List of Packages
            model_params: Model parameters

        Returns:
            Message
        """
        try:
            ## Print the starting message
            logging.info("Starting DRL\n")

            ## Initialize the ULD and packages
            ULD_DRL_ = ULD_DRL(ULD_data)
            PACKAGES_DRL_ = Package_DRL(Package_data)

            ## Initialize the environment
            env = Bin_Packing_Env(ULD_DRL_, PACKAGES_DRL_, ULD_data, Package_data)
            env.reset()

            ## Initialize the model
            if model_params["model_name"] == "A2C":
                model = A2C(
                    "MlpPolicy",
                    env,
                    ent_coef=model_params["ent_coef"],
                    learning_rate=model_params["learning_rate"],
                    verbose=2,
                    use_sde=model_params["use_sde"],
                    sde_sample_freq=model_params["sde_sample_freq"],
                    gamma=model_params["gamma"],
                )
            elif model_params["model_name"] == "PPO":
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    ent_coef=model_params["ent_coef"],
                    learning_rate=model_params["learning_rate"],
                    use_sde=model_params["use_sde"],
                    sde_sample_freq=model_params["sde_sample_freq"],
                    gamma=model_params["gamma"],
                )
            else:
                raise ValueError("Invalid model name")

            ## Train the model
            model.learn(
                total_timesteps=model_params["total_timesteps"],
                reset_num_timesteps=False,
                tb_log_name=model_params["model_name"],
            )

            ## Close the environment
            env.close()

            ## Print the ending message
            logging.info("Ending DRL\n")
        except Exception as e:
            logging.error(f"DRL Error: {e}")
            raise e
