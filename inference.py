import os
import json
import torch
import numpy as np
import importlib.util

import vllm
from vllm import LLM, SamplingParams

from copy import deepcopy
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
)
from datasets import load_dataset
from strictfire import StrictFire
from tqdm import tqdm
from utils.misc import get_logger, config
from utils.constants import INFERENCE_OUTPUT
from typing import Dict, Any, List, Literal
from time import time

package_name = "flash_attn"
spec = importlib.util.find_spec(package_name)
FLASH_AVAILABLE = True
if spec is None:
    FLASH_AVAILABLE = False

SEED = 111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def vllm_generate(model, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=1,
        max_tokens=512,
        top_p=1.0,
        include_stop_str_in_output=True,
    )
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]

    return outputs


def main(
    model_name: str,
    task_name: Literal[
        "refinement_single",
        "refinement_multi",
        "expansion_single",
        "expansion_multi",
        "follow-up_single",
        "follow-up_multi",
        "recollection_single_cls",
        "recollection_multiple_cls",
        "recollection_single_global-inst",
        "recollection_multi_global-inst",
        "cls_ablation_gold",
        "cls_ablation_dgc",
        "cls_ablation_sgc",
        "cls_ablation_rc",
    ],
    conv_key: str = "conv",
    system_message: str = "You are a helpful, respectful and honest assistant.",
    output_key: str = "gen_resp",
    resume: bool = False,
    use_gold_history: bool = False,
):
    logger = get_logger(
        name=__name__,
        console_level="info",
        file_level="debug",
        log_path=os.path.join(
            "log",
            f"{task_name}_{model_name}.log",
        ),
        maxBytes=10000000,
    )
    data = load_dataset("wckwan/MT-Eval", task_name, split="test").to_list()
    prompts = []
    for i, row in enumerate(data):
        # create prompt
        conv = deepcopy(config[model_name]["chat_template"])
        if system_message:
            conv.set_system_message(system_message)
        for turn in row[conv_key]:
            conv.append_message(conv.roles[0], turn["user"])
            conv.append_message(conv.roles[1], turn["sys"])
            if not turn["do_inference"]:
                pbar.update(1)
                continue
            if resume and output_key in turn:
                pbar.update(1)
                if not use_gold_history:
                    conv.update_last_message(turn[output_key])
                continue
            conv.update_last_message(None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
        breakpoint()



    use_flash = config[model_name]["use_flash_attn"] and FLASH_AVAILABLE
    if use_flash:
        logger.info("Using flash attention2.")

    model = vllm.LLM(
        model="allenai/tulu-2-dpo-7b",
        tensor_parallel_size=torch.cuda.device_count(),
    )

    breakpoint()
    outputs = vllm_generate(model=model, prompts=prompts[:10])
    generations_dir = "/net/nfs.cirrascale/mosaic/nouhad/projects/MT-Eval-fork/inference_outputs/refinement"
    out_file_path = os.path.join(generations_dir, f"preds.json")
    with open(out_file_path, 'w') as f_out:
        json.dump(outputs, f_out, indent=4)

    logger.info(f"Finished running. Output saved in {out_file_path}.")


if __name__ == "__main__":
    StrictFire(main)
