SPEED = 20.0
NEIGHBORS = 10
HORIZON = 10


from itertools import product
#betas = [0.01, 0.10, 0.25]
#ego_pars = [1, 5, 10]
#prod_vars = [betas, ego_pars, ego_pars]
#combinations = product(*prod_vars) # cartesian product
# - Horizon: 10
# - Max Speed: 20
# - Risk: [0.01, 0.1, 0.2, 0.3]
# - ego\_x\_param/ego\_y\_param: {1:10, 1:4, 1:2, 1:1, 2:1, 4:1, 10:1}
# - Neighbors: 10
# - Control damping: 3
betas = [0.1, 0.18, 0.20, 0.22, 0.3]
x = [1, 4, 7, 10]
y = [1, 4, 7, 10]
#xy = [(1, 10), (1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (10, 1)]
prod_vars = [betas, x, y]
combinations = product(*prod_vars)
print(f'Number of combinations: {len(betas) * len(x) * len(y)}')

import os
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '../../../nuplan/dataset')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '../../../nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '../../../nuplan/dataset/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.worker_pool_builder import build_worker

### set up config file - edit in the config access to change ###
import hydra
CONFIG_PATH = "../../nuplan/planning/script/config/common/myconfigs"
CONFIG_NAME = "scenario_access"
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[])
###

### create all scenario objects as specified in config file ###
scenario_builder = build_scenario_builder(cfg)
scenario_filter = build_scenario_filter(cfg.scenario_filter)
worker = build_worker(cfg)
scenarios = scenario_builder.get_scenarios(scenario_filter, worker) # List[AbstractScenario]

scidx = 22
scenario = scenarios[scidx]
scenario_token = scenario.token
print(f'Scenario token: {scenario_token}')


EXP_NAME = 'grid_search_over_beta_and_C_risk2'

from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths
BASE_CONFIG_PATH = os.path.join(os.getenv('NUPLAN_TUTORIAL_PATH', ''), '../../nuplan/planning/script')
simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
    'scenario_filter=one_of_each_scenario_type',  # simulate only one log
    #"scenario_filter.log_names=['2021.06.09.14.58.55_veh-35_01894_02311']",
    "scenario_filter.scenario_tokens=['" + scenario_token +"']",  # use 2 total scenarios
]

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = '/home/nishka/nuplan-devkit/tutorials/saved_simulations'

# Select simulation parameters
EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

from nuplan.planning.script.run_simulation import run_simulation as main_simulation
from nuplanlqrplanner import NuPlanLQRPlanner

for comb in combinations:
    beta, ego_x, ego_y = comb
    
    # change job name every combination
    job_name = f'sc{scidx}_b{beta}_ego_x{ego_x}_y{ego_y}'

    print(f'RUNNING JOB_NAME: {job_name}')
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=simulation_hydra_paths.config_path)
    
    # Compose the configuration
    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=[
        f'group={SAVE_DIR}',
        f'experiment_name={EXP_NAME}',
        f'job_name={job_name}',
        'experiment=${experiment_name}/${job_name}',
        'worker=sequential',
        f'ego_controller={EGO_CONTROLLER}',
        f'observation={OBSERVATION}',
        f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}, "../nuplan/planning/script/config/simulation"]',
        'output_dir=${group}/${experiment}',
        'seed=3',
        *DATASET_PARAMS,
    ])
    
    planner = NuPlanLQRPlanner(scenarios[scidx], horizon=HORIZON, speed=SPEED, neighbors=NEIGHBORS,
                               risk=beta, ego_x_par=ego_x, ego_y_par=ego_y)
    main_simulation(cfg, planner)