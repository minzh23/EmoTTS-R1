import hydra
import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional
from generate_tts_batch import main as inference
from tts_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, DecodeConfig


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )
    output_text_only: bool = field(
        default=False, metadata={"help": "Decode text only"}
    )
    speech_sample_rate: int = field(
        default=24000, metadata={"help": "The sample rate for speech"}
    )
    audio_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to audio prompt"}
    )
    multi_round: bool = field(
        default=False, metadata={"help": "Multi-round inference"}
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    inference(cfg)

if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # Execution will pause here until a debugger connects

    # print("Debugger attached. Continuing execution...")
    main_hydra()
