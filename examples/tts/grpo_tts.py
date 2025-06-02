# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import logging
import hydra
from openai import OpenAI
import os
import base64
import wandb

from datasets import load_dataset, load_from_disk

from grpo_trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf

from tts_config import *

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset

os.environ["OPENAI_API_KEY"] = "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"

def accuracy_reward(prompts, audio_outputs, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def convert_to_base64(audio_sample_path): 
        with open(audio_sample_path, "rb") as audio_file:
            audio_data = audio_file.read()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
        return base64_audio
    
    client = OpenAI()
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []


    for prompt, audio_output in zip(prompts, audio_outputs):
    
        
        base64_audio = convert_to_base64(audio_output)
        source_prompt = prompt["source_prompt"]
        emotion_prompt = prompt["emotion_prompt"]
        completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {"role": "developer", "content": "You are a strict assistant who tries to provide precise evaluation. You are given a text message and an audio input. In the text message, there is a target text and a emotional prompt. You have to decide how the input audio conveyed the emotion described in the emotional prompt. Rate the audio from 0 to 10, where 0 is the worst and 10 is the best. Output only the rating number between <answer> and </answer>, with no further explanation. "},
            {"role": "user", 
            "content": [
                {"type": "text", "text": f"The text message is: {source_prompt}. The emotional prompt is : {emotion_prompt}"},
                {
                    "type": "input_audio", 
                    "input_audio":
                    {
                        "data": base64_audio,
                        "format": "wav"
                    }
                }
            ]
            }
        ]
        )
        reward_str = extract_answer(completion.choices[0].message.content)
        if reward_str == "": 
            print(f"Invalid reward")
            reward = 1.0
        else: 
            reward = float(reward_str) / 10.0
            reward = max(0.0, min(1.0, reward))
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                # f.write(f"Content: {content}\n")
                # f.write(f"Solution: {sol}\n")
            
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
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

    main(cfg)

def main(kwargs: DictConfig):
    
    train_config, fsdp_config, model_config, log_config, dataset_config, decode_config = kwargs.train_config, kwargs.fsdp_config, kwargs.model_config, kwargs.log_config,kwargs.dataset_config, kwargs.decode_config
    OmegaConf.set_struct(kwargs,False)
    del kwargs["train_config"]
    del kwargs["fsdp_config"]
    del kwargs["model_config"]
    del kwargs["log_config"]
    del kwargs["dataset_config"]
    del kwargs["decode_config"]
    OmegaConf.set_struct(kwargs,True)
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in ["accuracy"]]

    	# Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
    	level=logging.INFO, 
    	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    	datefmt="%Y-%m-%d %H:%M:%S",
    	filemode='w'
    )
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 
    logger.addHandler(file_handler)

    logger.info("train_config: {}".format(train_config))
    logger.info("fsdp_config: {}".format(fsdp_config))
    logger.info("model_config: {}".format(model_config))

    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)

    logger.info("dataset_config: {}".format(dataset_config))
    dataset = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        script_args=train_config,
        model_config=model_config,
        decode_config=decode_config,
        vocab_config=model_config.vocab_config,
        # script_args=script_args,
        train_dataset=dataset,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        # processing_class=Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",padding_side="right"),
        # peft_config=get_peft_config(model_args),
        # attn_implementation=model_config.attn_implementation,
        kwargs=kwargs, 
    )
    
    # if train_config.resume_from_checkpoint is not None:
    #     checkpoint = train_config.resume_from_checkpoint
    #     trainer.train(resume_from_checkpoint=checkpoint)
    # else:
    wandb.init(
        project="EmoTTS-R1",
        name="grpo",
        config=OmegaConf.to_container(train_config, resolve=True),
        mode="online",
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(train_config.output_dir)
    if train_config.push_to_hub:
        trainer.push_to_hub(dataset_name=dataset_config.dataset_name)


if __name__ == "__main__":
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # if local_rank == 0:
    #     import debugpy
    #     debugpy.listen(("127.0.0.1", 5678))
    #     print("Waiting for debugger to attach...")
    #     debugpy.wait_for_client()
    #     print("Debugger attached, starting execution...")
    main_hydra()                     