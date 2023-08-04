from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.worker_pool_builder import build_worker

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="/home/nishka/nuplan-devkit/nuplan/planning/script/config/common/myconfigs", config_name="scenario_access")
def run(cfg: DictConfig):
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    
    worker = build_worker(cfg)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)
    print(len(scenarios))
   
if __name__ == '__main__':
    run()