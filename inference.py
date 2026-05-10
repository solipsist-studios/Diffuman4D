import os
import sys
import gc

# CRITICAL: Set PyTorch memory allocator to use expandable segments
# max_split_size_mb: prevents tiny fragments by not splitting below this size
# expandable_segments: allows reusing fragmented memory more efficiently
# This fixes CUDA OOM due to memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Now do all other imports
from typing import Any, Dict, Optional, Tuple

import torch
import hydra
from glob import glob
from omegaconf import DictConfig

from src.samplers.sampling_runner import SamplingRunner
from src.utils import RankedLogger, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)

# for easier debugging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def inference(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Verify PyTorch memory allocator configuration
    if torch.cuda.is_available():
        log.info(f"CUDA available: {torch.cuda.device_count()} devices")
        log.info(f"PYTORCH_CUDA_ALLOC_CONF env var: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
        
        # The best way to verify is to check the env var is set and show memory stats
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        log.info(f"GPU 0: {mem_free / 1024**3:.2f}GB free / {mem_total / 1024**3:.2f}GB total")

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating pipelines <{cfg.model._target_}>")
    pipelines = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating sampler <{cfg.sampler._target_}>")
    sampler = hydra.utils.instantiate(cfg.sampler, dataset=dataset, pipelines=pipelines)
    runner = SamplingRunner(sampler)

    if cfg.sampling:
        log.info("Sampling...")
        runner.inference()

    if cfg.to_nerfstudio:
        log.info("Converting results to nerfstudio format...")
        runner.to_nerfstudio()

    if cfg.evaluating:
        log.info("Evaluating results...")
        runner.evaluate()


@hydra.main(version_base="1.3", config_path="configs", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point"""
    print_config_tree(cfg, resolve=True, save_to_file=True)

    inference(cfg)


if __name__ == "__main__":
    main()
