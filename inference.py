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


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][-15:])
        return any([stop in tokens[-len(stop) :] for stop in self.stops])


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

def llama_generate(
    prompt,
    model,
    tokenizer,
    debug: bool = False,
    end_tokens: List[str] = [],
    **kwargs,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if tokenizer.eos_token:
        end_tokens.append(tokenizer.eos_token)
    stopping_criteria = StoppingCriteriaSub(tokenizer, end_tokens)
    if not debug:
        input_ids = input_ids.to(DEVICE)
    if debug:
        output = "some dummy text."
        return output, input_ids.shape[1]
    else:
        with torch.no_grad():
            start_time = time()
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                **kwargs,
            )
            used_time = time() - start_time
    s = generation_output.sequences[0]
    output_tokens = s[input_ids.shape[1] :]
    num_output_tokens = len(output_tokens)
    output = tokenizer.decode(output_tokens)
    for stop_token in end_tokens:
        output = output.replace(stop_token, "")
    return output, input_ids.shape[1], num_output_tokens / used_time


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
    load_8bit: bool = False,
    temperature: float = 1.0,
    top_p: float = 1,
    top_k: int = 50,
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    load_model_args: Dict[str, Any] = {},
    end_tokens: List[str] = [],
    resume: bool = False,
    use_gold_history: bool = False,
    n_forward: int = -1,
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

    model_path = config[model_name]["path"]
    use_flash = config[model_name]["use_flash_attn"] and FLASH_AVAILABLE
    if use_flash:
        logger.info("Using flash attention2.")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     load_in_8bit=load_8bit,
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True,
    #     attn_implementation="flash_attention_2" if use_flash else None,
    #     **load_model_args,
    # )
    model = vllm.LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=torch.cuda.device_count(),
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path, trust_remote_code=True
    # )
    outputs = vllm_generate(model=model, prompts=prompts)
    generations_dir = "/net/nfs.cirrascale/mosaic/nouhad/projects/MT-Eval-fork/inference_outputs/refinement"
    out_file_path = os.path.join(generations_dir, f"preds.json")
    with open(out_file_path, 'w') as f_out:
        json.dump(outputs, f_out, indent=4)



    logger.info(f"Finished running. Output saved in {out_filename}.")


if __name__ == "__main__":
    StrictFire(main)
