import logging
from typing import Any, Optional

import torch
import wandb


class WandbTrainingLogger():
    def __init__(self,
                 api_key: str,
                 project: str,
                 run_name: str,
                 model: torch.nn.Module,
                 config: dict[str, Any],
                 resume: bool = False,
                 logger: Optional[logging.Logger] = None):
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self.api_key = api_key
        self.project = project
        self.run_name = run_name
        self.model = model
        self.config = config
        self.resume = resume
        self.step = 0
        self.initialized_wandb = False
        self.previous_wandb_max_step = -1
        self._initialize()

    def _initialize(self):
        wandb.login(key=self.api_key)
        if self.resume:
            wandb_run_id, self.previous_wandb_max_step = self._get_latest_run_id(self.project, self.run_name)
            if wandb_run_id is not None:
                try:
                    wandb.init(
                        id=wandb_run_id,
                        resume="must",
                        project=self.project,
                        name=self.run_name,
                        config=self.config
                    )
                    self.initialized_wandb = True
                except Exception as e:
                    self._logger.error(f"Error initializing wandb: {str(e)}")
                    self.initialized_wandb = False
        if not self.initialized_wandb:
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config
            )
        wandb.watch(self.model, log="all", log_freq=100)

    def _get_latest_run_id(self, project: str, run_name: str, entity=None):
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path=path, filters={"display_name": run_name})
        if len(runs) > 0:
            return runs[0].id, runs[0].lastHistoryStep
        return None, 0

    def set_step(self, step: int) -> None:
        self.step = step
    
    def log(self, metrics: dict[str, Any]) -> None:
        if self.step < self.previous_wandb_max_step:
            self._logger.debug(f"Skipping logging step {self.step} because it is less than the previous max step {self.previous_wandb_max_step}")
            return
        wandb.log(metrics, step=self.step)

class DummyWandbLogger():
    def set_step(self, step: int) -> None:
        pass
    
    def log(self, metrics: dict[str, Any]) -> None:
        pass