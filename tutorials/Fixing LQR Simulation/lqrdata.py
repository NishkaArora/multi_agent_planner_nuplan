# Imports for LQRData
import torch
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# Imports for LQRData BFS
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch

from typing import List, Tuple
from shapely import LineString, Point
import shapely
import math
import numpy as np


class LQRData:
    def __init__(self, scenario, curr_state, speed, neighbors=-1, start_iter=0, horizon=-1):
        
        self.s = scenario
        self.data = {} # to be populated and used for LQRPlanner
        
        self.duration = self.s.duration_s
        self.num_frames = self.s.get_number_of_iterations() - 1
        self.scenario_type = self.s.scenario_type

        self.curr_state = curr_state
        self.start_iter = start_iter
        self.closest_line_point = (0, 0)

        self.car_speed = speed #m/s --- find this from history if possible
        
        #self.start_iter = self.get_closest_iter(curr_pos) # find start_iter using curr_pos

        #with open('start_iter_sim.txt', 'a') as f:
        #    f.write(str(self.start_iter))
        #    f.write('\n')
        
        if horizon == -1:
            self.horizon = self.num_frames
        else:
            self.horizon = horizon

        #if self.start_iter + self.horizon > self.num_frames:
        #    self.horizon = self.num_frames - self.start_iter + 1
        
        self.neighbors = neighbors
        
        #print(f'\nstart_iter:{self.start_iter}, horizon:{self.horizon}')

        # Variables for reference path BFS
        self.route_roadblocks = []
        self.candidate_lane_edge_ids = []
        #self.init = self.s.get_ego_state_at_iteration(self.start_iter).center
        #self.fin = self.s.get_ego_state_at_iteration(self.start_iter + self.horizon - 1).center
    
    # def get_closest_iter(self, curr_pos):
    #     min_dist = math.inf
    #     min_iter = 0
    #     for it in range(self.num_frames):
    #         dist = self.s.get_ego_state_at_iteration(it).center.distance_to(curr_pos.center)
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_iter = it
    #     return min_iter


    def populate_data(self):
        self.data['current_pos_gb'] = self.get_current_ego_pos_gb()
        self.data['current_pos'] = self.get_current_ego_pos()
        
        self.data['future_pos'] = self.get_future_ego_pos()
        #self.data['future_neighbor_pos'] = self.get_future_neighbor_pos_padded()
        
        self.data['current_lane'] = self.get_current_lane()

        self.data['future_neighbor_pos'] = self.get_n_closest_neighbors()
        self.data['current_neighbor_pos'] = self.data['future_neighbor_pos'][:, 0, :]
        
        #self.data['current_lane_global'] = self.data['current_lane']
        
        self.change_frame_to_ego_start()
        self.add_batch_size_dimension()
        #self.show_dims()

    # Simple helpers
    def add_batch_size_dimension(self):
        for key in self.data:
            self.data[key] = torch.unsqueeze(self.data[key], 0)

    def change_frame_to_ego_start(self):
        #print(f'Centered at Ego Start Pos: {str((self.curr_state[0], self.curr_state[1]))}') # curr_state[2] is orientation
        
        # use first and 5th point in current_lane to approximate initial instantaneous slope
        if self.data['current_lane'].shape[1] > 5:
            cur_lane_points = self.data['current_lane'][(0, 5), :]
            cur_lane_heading = math.atan2((cur_lane_points[1, 1] - cur_lane_points[0, 1]), (cur_lane_points[1, 0] - cur_lane_points[0, 0]))
        else:
            cur_lane_heading = 0 # deal with this later
        
        self.data['current_pos_gb'] = self.data['current_pos_gb']
        self.data['current_pos'] = self.data['current_pos']
        
        self.data['future_pos'] = torch.Tensor(self.convert_global_coords_to_local(self.data['future_pos'],
                                                                      (self.curr_state[0], self.curr_state[1], 0),
                                                                      -self.curr_state[2]))
        
        self.data['current_lane'] = torch.Tensor(self.convert_global_coords_to_local(self.data['current_lane'],
                                                                                     (self.curr_state[0], self.curr_state[1], 0), 
                                                                                     -self.curr_state[2]))
        
        self.data['current_neighbor_pos'] = torch.Tensor(self.convert_global_coords_to_local(self.data['current_neighbor_pos'],
                                                                                (self.curr_state[0], self.curr_state[1], 0),
                                                                                -self.curr_state[2]))
        
        # self.data['future_neighbor_pos'] = self.convert_global_coords_to_local(self.data['future_neighbor_pos'],
        #                                                                        (self.curr_state[0], self.curr_state[1], 0),
        #                                                                        self.curr_state[2])

    def convert_global_coords_to_local(self, coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   #rotation: Tuple[float, float, float, float]):
                                   rotation: float):
        """
        Converts global coordinates to coordinates in the frame given by the rotation quaternion and
        centered at the translation vector. The rotation is meant to be a z-axis rotation.
        :param coordinates: x,y locations. array of shape [n_steps, 2].
        :param translation: Tuple of (x, y, z) location that is the center of the new frame.
        :param rotation: Tuple representation of quaternion of new frame.
            Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
        :return: x,y locations in frame stored in array of share [n_times, 2].
        """
        #yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))
    
        transform = self.make_2d_rotation_matrix(angle_in_radians=rotation)
    
        coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T
    
        return np.dot(transform, coords).T[:, :2]

    def angle_of_rotation(self, yaw: float) -> float:
        """
        Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
        the yaw is aligned with the y axis (pi / 2).
        :param yaw: Radians. Output of quaternion_yaw function.
        :return: Angle in radians.
        """
        return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

    
    def make_2d_rotation_matrix(self, angle_in_radians: float) -> np.ndarray:
        """
        Makes rotation matrix to rotate point in x-y plane counterclockwise
        by angle_in_radians.
        """

        return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                         [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    def show_dims(self):
        for key in self.data:
            print(key + ': ' + str(self.data[key].shape))
    
    def get_xy_from_egostate(self, ego):    
        return ego.waypoint._oriented_box.center.x, ego.waypoint._oriented_box.center.y

    def get_xy_from_agent(self, agent):
        return (agent.center.x, agent.center.y)

    # Data-Getters from NuPlan Scenario
    def get_current_ego_pos_gb(self):
        #return torch.Tensor(self.get_xy_from_egostate(self.s.get_ego_state_at_iteration(self.start_iter)))
        return torch.Tensor((self.curr_state[0], self.curr_state[1]))

    def get_current_ego_pos(self):
        return torch.Tensor((0, 0))

    def get_future_ego_pos(self):
        future_ego_pos = []
        future_true = self.s.get_ego_future_trajectory(self.start_iter, self.horizon * 0.1, num_samples=self.horizon)
        for ego in future_true:
            future_ego_pos.append(self.get_xy_from_egostate(ego))
        return torch.Tensor(future_ego_pos)
    
    def get_n_closest_neighbors(self):
        # make sure self.current_lane is populated

        def euc_distance(x, y):
            return math.sqrt((y[1] - x[1]) ** 2 + (y[0] - x[0]) ** 2)

        closest_neighbors = np.zeros((self.neighbors, self.horizon, 2))
        
        # for each frame in 0 to self.horizon-1
        frame_id = 0
        for det_track in self.s.get_future_tracked_objects(self.start_iter, time_horizon=self.horizon * 0.1, num_samples=self.horizon):
            
            # for each tracked vehicle
            all_curr_neighbors = []
            for obj in det_track.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
                all_curr_neighbors.append(self.get_xy_from_agent(obj))
            
            cur_ref_point = self.data['current_lane'][frame_id, :]
            
            assert(len(all_curr_neighbors) >= self.neighbors)
            all_curr_neighbors = sorted(all_curr_neighbors, key=lambda pos: euc_distance(pos, cur_ref_point))
            closest_neighbors_frame = all_curr_neighbors[:self.neighbors]

            closest_neighbors[:, frame_id, :] = closest_neighbors_frame
            frame_id += 1
        return torch.Tensor(closest_neighbors)


    def get_future_neighbor_pos_padded(self):
        veh = {}
        # for each frame
        for det_track in self.s.get_future_tracked_objects(self.start_iter, time_horizon=self.horizon * 0.1, num_samples=self.horizon):
            # for each tracked vehicle
            for obj in det_track.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
                veh[obj.track_token] = [(None, None)] * self.horizon

        # populate known vehicle positions
        frame_id = 0
        for det_track in self.s.get_future_tracked_objects(self.start_iter, time_horizon=self.horizon * 0.1, num_samples=self.horizon):
            # for each tracked vehicle
            for obj in det_track.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
                veh[obj.track_token][frame_id] = self.get_xy_from_agent(obj)
            frame_id += 1
        if self.neighbors == -1:
            n_veh = list(veh.values())
        else:
            n_veh = list(veh.values())[:self.n]
        padded_n_veh = []
        for veh in n_veh:
            padded_n_veh.append(self.pad_both_sides(veh))
            
        return torch.Tensor(padded_n_veh)#[:, self.start_iter:self.start_iter+self.horizon, :]

    def pad_both_sides(self, arr):
        # forward
        last = (None, None)
        for idx in range(len(arr)):
            if arr[idx] != (None, None):
                last = arr[idx]
            else:
                arr[idx] = last
        
        # no values in array
        if last == (None, None):
            return -1
            
        # backward
        last = (None, None)
        for idx in range(len(arr)-1, -1, -1):
            if arr[idx] != (None, None):
                last = arr[idx]
            else:
                arr[idx] = last
        return arr

        
    def interpolate_to_dim(self, ref_path):
        #print(len(ref_path))
        line = LineString(ref_path)
        num_pts = self.horizon
        #num_pts = s0.data['future_pos'].shape[0]
        #num_pts = self.s.get_number_of_iterations() - 1
        #num_pts = self.num_frames
        distances = np.linspace(0, line.length, num_pts)
        points = [line.interpolate(distance) for distance in distances]
        return [(pt.x, pt.y) for pt in points]
        
        
    def trim_ref_path(self, full_bfs):
        
        init_dist = float('inf')
        fin_dist = float('inf')

        init_id = 0
        fin_id = 0
        
        for idx in range(len(full_bfs)):
            if init_dist > full_bfs[idx].distance_to(self.init):
                init_dist = full_bfs[idx].distance_to(self.init)
                init_id = idx
            if fin_dist > full_bfs[idx].distance_to(self.fin):
                fin_dist = full_bfs[idx].distance_to(self.fin)
                fin_id = idx
        
        ref_path = full_bfs[init_id:fin_id+1]
        return [(pt.x, pt.y) for pt in ref_path] #, [pt.y for pt in ref_path]

    def get_current_lane(self):
        # get current lane info
        bfs_output = self.get_bfs_path()
        
        full_bfs = []
        for lane in bfs_output:
            full_bfs += lane.baseline_path.discrete_path
        
        ref_line = LineString([[state.x, state.y] for state in full_bfs])
        closest_point = ref_line.project(Point(self.curr_state[0], self.curr_state[1]))
        closest_line_point = ref_line.interpolate(closest_point)
        self.closest_line_point = closest_line_point.x - self.curr_state[0], closest_line_point.y - self.curr_state[1]

        dist_horizon = self.horizon * self.car_speed * 0.1
        distances = np.linspace(0, dist_horizon, self.horizon)
        points = [ref_line.interpolate(closest_point + distance) for distance in distances]

        return torch.Tensor([[point.x, point.y] for point in points])
        # ref_path = self.trim_ref_path(full_bfs)

        # if len(ref_path) > 1:
        #     return torch.Tensor(self.interpolate_to_dim(ref_path))
        # else:
        #     return torch.Tensor(self.interpolate_to_dim(full_bfs))

    def get_bfs_path(self):
        # get route plan from map
        route_roadblocks_ids = self.s.get_route_roadblock_ids()

        self.route_roadblocks = []
        for id_ in route_roadblocks_ids:
            block = self.s.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.s.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self.route_roadblocks.append(block)

        # candidate lane IDs
        self.candidate_lane_edge_ids = [edge.id for block in self.route_roadblocks 
                                   if block for edge in block.interior_edges]
        
        current_lane_path, found = self.bfs(self.s.initial_ego_state)
        # assert(found == True)
        return current_lane_path

    def bfs(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        starting_edge = self.get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self.candidate_lane_edge_ids)
        offset = 1 if starting_edge.get_roadblock_id() == self.route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(self.route_roadblocks[-1], len(self.route_roadblocks[offset:]))
        if not path_found:
            print('BFS: Path not found')
        return route_plan, path_found
    
    def get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        starting_edge = None
        closest_dist = math.inf
        found_on_roadblock = False
        for edge in self.route_roadblocks[0].interior_edges + self.route_roadblocks[1].interior_edges:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                found_on_roadblock = True
                break
            # case ego does not start on a road block
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_dist:
                starting_edge = edge
                closest_dist = distance

        if not found_on_roadblock:
            print('Picking closest approx for starting roadblock')
        return starting_edge
    