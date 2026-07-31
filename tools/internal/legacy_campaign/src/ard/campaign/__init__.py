"""Durable, static single-host campaign orchestration."""

from .schema import CampaignSpec, JobSpec, bind_git_sha, effective_wandb_run_id, load_campaign
from .state import CampaignStateStore, JobState
from .worker import CampaignWorker

__all__ = [
    "CampaignSpec",
    "JobSpec",
    "CampaignStateStore",
    "JobState",
    "CampaignWorker",
    "bind_git_sha",
    "effective_wandb_run_id",
    "load_campaign",
]
