from typing import List, Type
import itertools
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from lqr.lqr import LQRPlanner
from lqrdata import LQRData
import math
import numpy as np
import pickle

filehandler = open(f'saved_data_from_orientation_fixed_lqr_sc{17}_horizon20_speed{5}_withinputs', 'wb')

class NuPlanLQRPlanner(AbstractPlanner):
    
    requires_scenario: bool = True
    
    
    def __init__(self, scenario: AbstractScenario, horizon, speed, scidx):
        """
        Constructor of NuPlanLQRPlanner.
        :param scenario: The scenario the planner is running on.
        :param horizon: The number of frames to predict in the future
        """
        self._scenario = scenario
        self.speed = speed
        self.i = 0
        #self._num_poses = num_poses
        #self._future_time_horizon = future_time_horizon
        self._horizon = horizon
        self._trajectory: Optional[AbstractTrajectory] = None


        self.save_data = {}

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def get_heading(self, traj):
        # Somehow confirm if this is correct
        heading = []
        h = 0.0
        for i in range(len(traj) - 1):
            pt1, pt2 = traj[i], traj[i+1]
            h = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            heading.append(h)
        heading.append(h)
        return heading
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        buffer = 1 # buffer for lqr zeros in the beginning (idk a better way around this)
        #current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        #curr_state = current_input.history.ego_state_buffer[-1] # last ego state
        
        curr_state, observations = current_input.history.current_state
        start = (curr_state.waypoint._oriented_box.center.x,
                 curr_state.waypoint._oriented_box.center.y,
                 curr_state.waypoint._oriented_box.center.heading)
        
        inputs = {}
        inputs['scenario'] = self._scenario
        inputs['curr_pos'] = start
        inputs['speed'] = 7
        inputs['neighbors'] = 4
        inputs['horizon'] = self._horizon + buffer
        
        s0 = LQRData(self._scenario, curr_state=start, speed=self.speed, neighbors=4, horizon=self._horizon+buffer)
        #print(f'\nProgress: {current_input.iteration.index}/{s0.num_frames}')
        s0.populate_data()

        lqrplanner = LQRPlanner(horizon=s0.horizon, risk = 0.0) # risk = 0 just follows reference
        traj = lqrplanner.forward(s0.data)['traj'].detach().numpy()[0]
        
        # Save s0 and traj
        save_iter = {}
        save_iter['data'] = s0
        save_iter['traj'] = traj
        save_iter['inputs'] = inputs
        self.save_data[current_input.iteration.index] = save_iter
        
        if current_input.iteration.index == 148:
            pickle.dump(self.save_data, filehandler)

        heading = self.get_heading(traj)
        traj_xyh = np.array([[pt[0], pt[1], h] for pt, h in zip(traj, heading)])
        abs_states = [StateSE2.deserialize(pose) for pose in traj_xyh]
        rel_states = absolute_to_relative_poses(abs_states)
        predictions = np.array([[pose.x, pose.y, pose.heading] for pose in rel_states])[buffer:]
        
        # with open('trajs_lqrplanner.txt', 'a') as f:
        #     f.write(str(predictions))
        #     f.write('\n\n')
        
        print('LQR Output Trajectory Length: ', len(traj_xyh))
        ego_history = current_input.history.ego_states
        
        print('Horizon: ', s0.horizon)
        states = transform_predictions_to_states(predictions, ego_history, round((s0.horizon - buffer) * 0.1, 3), 0.1)
        trajectory = InterpolatedTrajectory(states)
        #self._trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))
        return trajectory #self._trajectory
    
    def get_xy_from_egostate(self, ego):    
        return ego.waypoint._oriented_box.center.x, ego.waypoint._oriented_box.center.y