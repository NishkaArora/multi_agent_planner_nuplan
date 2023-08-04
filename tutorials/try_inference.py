import torch
import pickle
from inference import call_inference

from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap

if __name__ == "__main__":

    with open('examples/agents', 'rb') as pick:
        agents = pickle.load(pick)
        agents_obj = Agents(ego=torch.Tensor(agents['ego']), agents=torch.Tensor(agents['agents']))

    with open('examples/vector_map', 'rb') as pick:
        vmap = pickle.load(pick)
        vmap_obj = VectorMap(coords=torch.Tensor(vmap['coords']), 
                            lane_groupings=vmap['lane_groupings'],
                            multi_scale_connections=vmap['multi_scale_connections'],
                            on_route_status=vmap['on_route_status'],
                            traffic_light_data=vmap['traffic_light_data'])
    
    print(call_inference(vmap_obj, agents_obj))