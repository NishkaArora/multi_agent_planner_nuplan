from pathlib import Path
import os
import hydra

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper

from typing import cast
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


# Create a temporary directory to store the cache and experiment artifacts
# SAVE_DIR = Path(tempfile.gettempdir()) / 'tutorial_nuplan_framework'  # optionally replace with persistent dir
SAVE_DIR = Path('/home/nishka')
EXPERIMENT = 'training_logs/exp/training'
JOB_NAME = 'gpu0workerstrain_simple_vector_map_model'
LOG_DIR = SAVE_DIR / EXPERIMENT / JOB_NAME

# Get the checkpoint of the trained model
last_experiment = sorted(os.listdir(LOG_DIR))[-1]
train_experiment_dir = sorted(LOG_DIR.iterdir())[-1]
checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]
MODEL_PATH = str(checkpoint)
MODEL_PATH_cfg = str(checkpoint).replace("=", "\=")

# Name of the experiment
EXPERIMENT = 'Simple_VMM_Experiment'

### Not used ###
CHALLENGE = 'open_loop_boxes'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
]
# Location of path with all simulation configs
CONFIG_PATH = '../nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'
##################

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'experiment_name={EXPERIMENT}',
    f'group={SAVE_DIR}',
    'planner=ml_planner',
    'model=simple_vector_model',
    'planner.ml_planner.model_config=${model}',  # hydra notation to select model config
    f'planner.ml_planner.checkpoint_path={MODEL_PATH_cfg}',  # this path can be replaced by the checkpoint of the model trained in the previous section
    f'+simulation={CHALLENGE}',
    *DATASET_PARAMS,
])
planner_cfg = cfg.planner.ml_planner

torch_module_wrapper = build_torch_module_wrapper(planner_cfg.model_config)
model = LightningModuleWrapper.load_from_checkpoint(MODEL_PATH, model=torch_module_wrapper).model

# with open('examples/trajectory', 'rb') as pick:
#     traj = pickle.load(pick)

def call_inference(vmap_obj, agents_obj):
    model.eval()

    features = {'vector_map': vmap_obj, 'agents': agents_obj}

    return model.forward(features)