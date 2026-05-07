"""Evaluation script."""

import copy
import numpy as np
import os
import pickle
import re
import subprocess
import json
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from datasets import load_dataset
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from synth_metrics import synth_engineering_relaxed_match

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

# Suppress specific warning
import warnings

warnings.filterwarnings("ignore", message="The following generation flags are not valid")

# ---------------------------------------------------------------------------
# Optional reference evaluation profile with fixed generation defaults.
# Use `--eval_profile llm_jepa_official` to force these values regardless of
# any per-dataset caps passed by wrappers.
# ---------------------------------------------------------------------------
LLM_JEPA_OFFICIAL_MAX_NEW_TOKENS = 128
LLM_JEPA_OFFICIAL_MAX_LENGTH = 512

VALID_EVAL_PROFILES = ("fork", "llm_jepa_official")


def apply_eval_profile(args: argparse.Namespace) -> None:
    """Mutate generation-related fields on `args` when matching a fixed reference profile."""
    profile = getattr(args, "eval_profile", "fork")
    if profile == "llm_jepa_official":
        args.max_length = LLM_JEPA_OFFICIAL_MAX_LENGTH
        args.max_new_tokens = LLM_JEPA_OFFICIAL_MAX_NEW_TOKENS
        print(
            "[eval_profile=llm_jepa_official] "
            f"max_new_tokens={args.max_new_tokens}, max_length={args.max_length} "
            "(reference defaults; shell caps ignored).",
            flush=True,
        )
    elif profile != "fork":
        raise ValueError(
            f"eval_profile must be one of {VALID_EVAL_PROFILES}; got {profile!r}"
        )


# def use_llama_3_2_chat_template(tokenizer):
#     llama_3_2_chat_template = """{{- bos_token }}
# {%- if custom_tools is defined %}
#     {%- set tools = custom_tools %}
# {%- endif %}
# {%- if not tools_in_user_message is defined %}
#     {%- set tools_in_user_message = true %}
# {%- endif %}
# {%- if not date_string is defined %}
#     {%- if strftime_now is defined %}
#         {%- set date_string = strftime_now("%d %b %Y") %}
#     {%- else %}
#         {%- set date_string = "26 Jul 2024" %}
#     {%- endif %}
# {%- endif %}
# {%- if not tools is defined %}
#     {%- set tools = none %}
# {%- endif %}

# {#- This block extracts the system message, so we can slot it into the right place. #}
# {%- if messages[0]['role'] == 'system' %}
#     {%- set system_message = messages[0]['content']|trim %}
#     {%- set messages = messages[1:] %}
# {%- else %}
#     {%- set system_message = "" %}
# {%- endif %}

# {#- System message #}
# {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
# {%- if tools is not none %}
#     {{- "Environment: ipython\n" }}
# {%- endif %}
# {{- "Cutting Knowledge Date: December 2023\n" }}
# {{- "Today Date: " + date_string + "\n\n" }}
# {%- if tools is not none and not tools_in_user_message %}
#     {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
# {%- endif %}
# {{- system_message }}
# {{- "<|eot_id|>" }}

# {#- Custom tools are passed in a user message with some extra guidance #}
# {%- if tools_in_user_message and not tools is none %}
#     {#- Extract the first user message so we can plug it in here #}
#     {%- if messages | length != 0 %}
#         {%- set first_user_message = messages[0]['content']|trim %}
#         {%- set messages = messages[1:] %}
#     {%- else %}
#         {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
# {%- endif %}
#     {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
#     {{- "Given the following functions, please respond with a JSON for a function call " }}
#     {{- "with its proper arguments that best answers the given prompt.\n\n" }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
#     {{- first_user_message + "<|eot_id|>"}}
# {%- endif %}

# {%- for message in messages %}
#     {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
#         {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
#     {%- elif 'tool_calls' in message %}
#         {%- if not message.tool_calls|length == 1 %}
#             {{- raise_exception("This model only supports single tool-calls at once!") }}
#         {%- endif %}
#         {%- set tool_call = message.tool_calls[0].function %}
#         {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
#         {{- '{"name": "' + tool_call.name + '", ' }}
#         {{- '"parameters": ' }}
#         {{- tool_call.arguments | tojson }}
#         {{- "}" }}
#         {{- "<|eot_id|>" }}
#     {%- elif message.role == "tool" or message.role == "ipython" %}
#         {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
#         {%- if message.content is mapping or message.content is iterable %}
#             {{- message.content | tojson }}
#         {%- else %}
#             {{- message.content }}
#         {%- endif %}
#         {{- "<|eot_id|>" }}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
# {%- endif %}
# """
#     if tokenizer.chat_template != llama_3_2_chat_template:
#         tokenizer.chat_template = llama_3_2_chat_template


def get_messages(model_name, messages):
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        return full_messages
    else:
        return messages


def get_user_messages(model_name, messages):
    return copy.deepcopy(messages)[1:2]


def get_assistant_messages(model_name, messages):
    if "google/gemma" in model_name:
        assistant_messages = copy.deepcopy(messages)[2:3]
        assistant_messages[0]["role"] = "user"
        return assistant_messages
    else:
        return messages[2:3]


def load_model_and_tokenizer(model_name, original_model_name, load_in_8bit=False, load_in_4bit=False, device_map="auto"):
    """Load model and tokenizer with optional quantization"""
    
    print(f"Loading model: {model_name}")
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    
    # Load tokenizer
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    special_tokens = ["<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>", "<|predictor_4|>", "<|predictor_5|>",
                      "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        # model.resize_token_embeddings(len(tokenizer))
        if torch.cuda.current_device() == 0:
            print(f"Added {len(new_tokens)} new special tokens")

    if "google/gemma" in original_model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if not quantization_config else None,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if not quantization_config else None,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
    
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded on device: {model.device if hasattr(model, 'device') else 'multi-device'}")
    return model, tokenizer


# def apply_chat_template_selector(model_name):

#     def apply_chat_template_llama3_eval(tokenizer, messages):
#         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     def apply_chat_template_gemma_eval(_, messages):
#         """Applies chat template for Gemma models"""
#         output = "<bos>"
#         need_header = True
#         for item in messages:
#             if need_header and (item["role"] == "system" or item["role"] == "user"):
#                 need_header = False
#                 output += "<start_of_turn>user\n"
#             if item["role"] == "system":
#                 output += item["content"] + "\n\n"
#             elif item["role"] == "user":
#                 output += item["content"] + "<end_of_turn>\n<start_of_turn>model\n"
#             elif item["role"] == "assistant":
#                 output += item["content"] + "<end_of_turn>\n"
        
#         return output

#     def apply_chat_template_llama2(_, messages):
#         """Applies chat template for Llama2 models"""
#         output = "<s>"
#         need_header = True
#         for item in messages:
#             if need_header and (item["role"] == "system" or item["role"] == "user"):
#                 need_header = False
#                 output += "[INST] "
#             if item["role"] == "system":
#                 output += "<<SYS>>\n" + item["content"] + "\n<</SYS>>\n\n"
#             elif item["role"] == "user":
#                 output += item["content"] + " [/INST]"
#             elif item["role"] == "assistant":
#                 output += " " + item["content"] + " </s>"
        
#         return output

#     def apply_chat_template_openelm_eval(_, messages):
#         """Appies chat template for OpenELM models."""
#         output = ""
#         for item in messages:
#             if item["role"] == "system":
#                 output += "### System:\n" + item["content"] + "\n\n"
#             elif item["role"] == "user":
#                 output += "### User:\n" + item["content"] + "\n\n"
#             elif item["role"] == "assistant":
#                 output += "### Assistant:\n" + item["content"] + "\n\n"
        
#         return output + "### Assistant:"
    
#     if "apple/OpenELM" in model_name or "microsoft/phi" in model_name:
#         return apply_chat_template_openelm_eval
#     elif "google/gemma" in model_name:
#         return apply_chat_template_gemma_eval
#     return apply_chat_template_llama3_eval


def format_conversation(messages, tokenizer, include_assistant=False, plain=False, similarity=False):
    """Format conversation for the model"""
    # Filter out assistant messages if we're generating them
    if not include_assistant:
        messages = [msg for msg in messages if msg['role'] != 'assistant']

    # Use chat template if available
    if plain:        
        if similarity:
            formatted_text = messages[0]["content"]
        else:
            formatted_text = messages[1]["content"] + "<|perception|>"
    else:
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)

    return formatted_text


def generate_response(model, tokenizer, prompt, generation_config, max_new_tokens, unmask_assistant_special_tokens=False):
    """Generate a single response"""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=generation_config.max_length,
        add_special_tokens=not unmask_assistant_special_tokens
    )
    
    # Move to model device
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # For multi-device setups, let the model handle device placement
        pass
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    
    # Decode only the generated part (exclude input)
    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up response
    response = response.strip()
    
    # Remove any trailing special tokens or formatting
    if response.endswith("<|end|>"):
        response = response[:-7].strip()
    
    return response


def get_sequence_embedding(model, tokenizer, prompt, generation_config, pooling='last', layer=-1, unmask_assistant_special_tokens=False):
    """Get sequence embedding"""
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=generation_config.max_length,
        add_special_tokens=not unmask_assistant_special_tokens
    )
    
    # Move to model device
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # For multi-device setups, let the model handle device placement
        pass
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract embeddings from last layer
    hidden_states = outputs.hidden_states[layer]  # Shape: [batch_size, seq_len, hidden_dim]
    
    if pooling == 'last':
        # Use last token embedding (common for decoder models)
        embedding = hidden_states[0, -1, :]
    elif pooling == 'mean':
        # Mean pooling over sequence
        attention_mask = inputs['attention_mask']
        embedding = (hidden_states[0] * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    elif pooling == 'cls':
        # Use first token (if model has CLS-like token)
        embedding = hidden_states[0, 0, :]
    
    return embedding


def split_dataset_and_save(input_file, train_file, test_file, test_size=0.2, seed=42):
    """Split dataset into train and test sets and save them"""
    
    print(f"\nSplitting dataset: {input_file}")
    print(f"Test size: {test_size}")
    print(f"Random seed: {seed}")
    
    # Load dataset
    if input_file.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=input_file)['train']
    else:
        raise ValueError("Only JSONL files are supported")
    
    print(f"Total examples: {len(dataset)}")
    
    # Split dataset
    split_data = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    train_dataset = split_data['train']
    test_dataset = split_data['test']
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    # Save train set
    print(f"Saving train set to: {train_file}")
    with open(train_file, 'w') as f:
        for example in train_dataset:
            f.write(json.dumps(example) + '\n')
    
    # Save test set
    print(f"Saving test set to: {test_file}")
    with open(test_file, 'w') as f:
        for example in test_dataset:
            f.write(json.dumps(example) + '\n')
    
    print("Dataset splitting complete!")
    return train_file, test_file


def relative_probability(model, tokenizer, prompt, max_length, unmask_assistant_special_tokens=False):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=not unmask_assistant_special_tokens
    )
    
    # Move to model device
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # For multi-device setups, let the model handle device placement
        pass
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    next_token_logits = logits[0, -1]
    token_A_id = tokenizer.convert_tokens_to_ids("A")
    token_B_id = tokenizer.convert_tokens_to_ids("B")
    token_C_id = tokenizer.convert_tokens_to_ids("C")
    token_D_id = tokenizer.convert_tokens_to_ids("D")
    probs = torch.softmax(next_token_logits, dim=-1)
    probs_tensor = torch.tensor([probs[token_A_id].item(), probs[token_B_id].item(), probs[token_C_id].item(), probs[token_D_id].item()])
    answers = ["A", "B", "C", "D"]
    return answers[torch.argmax(probs_tensor)]


spider_pattern = re.compile(r"For db_id:\[(.+)\]")


def spider_eval(generated, ground_truth, spider_path, debug=0):
    # Pass raw generated SQL
    # straight to sqlite3 and compare stdout == stdout. No _truncate_sql, no
    # empty-gt guard (the paper counts empty==empty as a match).
    #
    # We retain two *defensive* safety rails that do NOT change behavior when
    # spider_data is present and healthy:
    #   (a) missing-dbfile guard: prevents a silent 100% if someone forgets to
    #       unzip spider_data.zip (both subprocess calls would return "").
    #   (b) subprocess timeout: prevents a runaway query from hanging the sweep.
    db_id = re.search(spider_pattern, ground_truth[1]["content"])
    assert db_id
    db_id = db_id.group(1)
    dbfile = os.path.join(spider_path, db_id, db_id + '.sqlite')

    if not os.path.exists(dbfile):
        if debug == 1:
            print(f"[SPIDER] missing dbfile: {dbfile}")
        return False

    try:
        result = subprocess.run(["sqlite3", dbfile, generated],
                                capture_output=True, text=True, timeout=10)
        gen_result = result.stdout

        result = subprocess.run(["sqlite3", dbfile, ground_truth[2]["content"]],
                                capture_output=True, text=True, timeout=10)
        gt_result = result.stdout
    except Exception:
        return False

    if debug == 1:
        print("[GEN]", gen_result)
        print("[GT:]", gt_result)

    return gen_result == gt_result


gsm8k_pattern = re.compile(r"\n#### (.+)$")
# Relaxed pattern: accept any "#### <answer>" occurrence; we take the LAST one.
gsm8k_pattern_relaxed = re.compile(r"####\s*([^\s#]+)")


def _normalize_gsm8k_answer(s):
    """Strip commas, $, and trailing punctuation so '1,234.' matches '1234'."""
    if s is None:
        return None
    s = s.strip().rstrip(".").rstrip(",").rstrip(")")
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    return s


def _dataset_kind(input_file):
    """Identify dataset by basename so `datasets/gsm8k_test.jsonl` routes to the gsm8k branch."""
    base = os.path.basename(input_file or "").lower()
    for key in ("gsm8k", "spider", "nq_open", "rotten_tomatoes", "hellaswag", "synth", "turk"):
        if base.startswith(key):
            return key
    return "generic"


# Standard open-domain QA normalization (SQuAD / NQ-Open convention): lowercase,
# strip articles "a/an/the", strip punctuation, collapse whitespace. Used for
# nq_open's "gt appears in gen" containment check.
_NQ_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_NQ_PUNCT = re.compile(r"[^\w\s]")


def _normalize_nq_open(s):
    if s is None:
        return ""
    s = s.lower()
    s = _NQ_PUNCT.sub(" ", s)
    s = _NQ_ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def eval(generated, ground_truth, input_file, spider_path, startswith=False, debug=0):
    if startswith:
        if debug == 1:
            print("[GEN]", generated)
            print("[GT:]", ground_truth[2]["content"])
            print("-----startswith-----")
        return generated.startswith(ground_truth[2]["content"])

    kind = _dataset_kind(input_file)

    if kind == "gsm8k":
        # GT may end with a trailing newline / whitespace after `#### X`; generation may trail with
        # more reasoning after the answer. Use findall and take the LAST `#### X` occurrence.
        gt_matches = gsm8k_pattern_relaxed.findall(ground_truth[2]["content"])
        gen_matches = gsm8k_pattern_relaxed.findall(generated)
        gt_answer = _normalize_gsm8k_answer(gt_matches[-1]) if gt_matches else None
        gen_answer = _normalize_gsm8k_answer(gen_matches[-1]) if gen_matches else None
        if debug == 1:
            print("[RAW]", generated)
            print("[GEN]", gen_answer)
            print("[GT:]", gt_answer)
            print("-----GSM8K-----")
        if gt_answer is None:
            return False
        return gt_answer == gen_answer

    if kind == "spider":
        return spider_eval(generated, ground_truth, spider_path, debug=debug)

    if kind == "nq_open":
        # Standard open-domain QA: the gold short answer should appear in the
        # generation (after normalization: lowercase, strip articles + punct).
        # The previous direction (`gen-fragment in gt`) was backwards and
        # systematically under-counted correct answers like
        #   gen="The answer is Vancouver."  gt="Vancouver".
        # GT may contain multiple acceptable answers separated by "; ".
        gen_n = _normalize_nq_open(generated)
        gt_raw = ground_truth[2]["content"]
        candidates = [g.strip() for g in gt_raw.split("; ") if g.strip()] or [gt_raw]
        for cand in candidates:
            cand_n = _normalize_nq_open(cand)
            if cand_n and cand_n in gen_n:
                if debug == 1:
                    print("[GEN]", gen_n[:120])
                    print("[GT:]", cand_n)
                    print("-----nq_open HIT-----")
                return True
        if debug == 1:
            print("[GEN]", gen_n[:120])
            print("[GT:]", [_normalize_nq_open(c) for c in candidates])
            print("-----nq_open MISS-----")
        return False

    if kind == "hellaswag":
        # 4-way multiple choice. `relative_probability` (triggered by the
        # basename-based hellaswag router upstream) already returns one of
        # {"A","B","C","D"}. Ground truth is the single-letter answer.
        gt = ground_truth[2]["content"].strip().upper()
        gen = (generated or "").strip().upper()
        # Be defensive in case a free-form generator slipped through: take
        # the first letter in {A,B,C,D}.
        first = next((c for c in gen if c in "ABCD"), "")
        if debug == 1:
            print("[GEN]", first, "(from:", gen[:80], ")")
            print("[GT:]", gt)
            print("-----hellaswag-----")
        return first == gt

    if kind == "turk":
        gt = ground_truth[2]["content"]
        gen = (generated or "").strip()
        if debug == 1:
            print("[GEN]", gen)
            print("[GT:]", gt)
            print("-----turk-----")
        if gen == gt:
            return True
        if gen.startswith(gt):
            remainder = gen[len(gt):]
            if not remainder or not remainder[0].isalnum():
                return True
        return False

    if kind == "rotten_tomatoes":
        # Single-token label ("Good"/"Bad"). Model often starts with the label and keeps talking,
        # so compare only the first whitespace-separated token (case-insensitive, stripped of punct).
        gt = ground_truth[2]["content"].strip()
        gen = generated.strip()
        first = gen.split()[0] if gen else ""
        first = first.rstrip(".,!?:;\"'").strip()
        if debug == 1:
            print("[GEN]", first, "(from:", gen[:80], ")")
            print("[GT:]", gt)
            print("-----rotten_tomatoes-----")
        return first.lower() == gt.lower()

    if kind == "synth":
        # Exact string match on stripped generation and gold.
        gt = (ground_truth[2]["content"] or "").strip()
        gen = (generated or "").strip()
        if debug == 1:
            print("[GEN]", repr(gen))
            print("[GT:]", repr(gt))
            print("-----synth (strict)-----")
        return gen == gt

    # Default: generic / unknown JSONL. Lenient prefix match for models that
    # emit the correct answer then ramble (OpenELM/Llama-2); not used for synth.
    gt = ground_truth[2]["content"]
    gen = (generated or "").strip()
    if debug == 1:
        print("[GEN]", gen)
        print("[GT:]", gt)
        print("-----")
    if gen == gt:
        return True
    if gen.startswith(gt):
        remainder = gen[len(gt):]
        if not remainder or not remainder[0].isalnum():
            return True
    return False


def process_dataset(input_file, output_file, original_model_name, model, tokenizer, 
                    generation_config, spider_path, max_examples=None, skip_existing=True,
                    split_tune_untune=False, start_index=1, layer=-1, pooling="last",
                    debug=0, similarity=False, startswith=False, max_new_tokens=128, t_sne=False,
                    plain=False, t_sne_type=None, model_name=None, unmask_assistant_special_tokens=False):
    """Process dataset and generate responses"""
    
    # Load dataset
    if input_file.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=input_file)['train']
    else:
        raise ValueError("Only JSONL files are supported")
    
    print(f"Loaded {len(dataset)} examples from {input_file}")
    if os.path.basename(input_file or "").lower().startswith("synth"):
        print(
            "[synth] Strict EM: stripped(gen)==stripped(gt). "
            "Also prints SYNTH engineering-relaxed rate (prefix+non-alnum boundary). "
            "Use --eval_profile llm_jepa_official for HF gen defaults (max_new_tokens=128, max_length=512).",
            flush=True,
        )

    # Limit examples if specified
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"Processing {len(dataset)} examples (limited by max_examples)")
    
    # Check if output file exists and load existing results
    existing_results = {}
    if not skip_existing and os.path.exists(output_file):
        print(f"Loading existing results from {output_file}")
        with open(output_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    existing_results[line_num] = json.loads(line.strip())
                except:
                    continue
        print(f"Found {len(existing_results)} existing results")
    
    assert start_index == 1
    # Process examples
    results = []
    failed_count = 0
    

    sim_list = []
    sim_list_startswith = []
    sim_list_untune = []

    synth_eng_hits = 0
    synth_eng_relaxed_only = 0
    synth_n = 0

    embedding_list = []
    label_list = []
    sample_list = []

    # apply_chat_template_func = apply_chat_template_selector(original_model_name)
    with open(output_file, 'w') as f:
        for idx, example in enumerate(tqdm(dataset, desc="Generating responses")):
            
            # Skip if already processed
            if not skip_existing and idx in existing_results:
                results.append(existing_results[idx])
                f.write(json.dumps(existing_results[idx]) + '\n')
                f.flush()
                continue
            
            try:
                # Get the conversation messages
                messages = example['messages']

                if similarity:
                    input = get_user_messages(original_model_name, messages)
                    input_prompt = format_conversation(input, tokenizer, similarity=similarity, plain=plain)
                    input_embedding = get_sequence_embedding(model, tokenizer, input_prompt, generation_config,
                                                             layer=layer, pooling=pooling, unmask_assistant_special_tokens=unmask_assistant_special_tokens)
                    output = get_assistant_messages(original_model_name, messages)
                    output_prompt = format_conversation(output, tokenizer, include_assistant=True, similarity=similarity, plain=plain)
                    output_embedding = get_sequence_embedding(model, tokenizer, output_prompt, generation_config,
                                                              layer=layer, pooling=pooling, unmask_assistant_special_tokens=unmask_assistant_special_tokens)

                    if debug == 3:
                        print(f"INPUT: {input_prompt}")
                        print(f"OUTPUT: {output_prompt}")

                    cos_sim = F.cosine_similarity(input_embedding, output_embedding, dim=-1).float().cpu()
                    if t_sne_type == 'in_n_out':
                        # To understand Enc(Text) and Enc(Code) relationship 
                        embedding_list.extend([input_embedding, output_embedding])
                        label_list.extend([0, 1])
                    elif t_sne_type == 'paraphrase':
                        # To understand Enc(Text) among paraphrase ID groups 
                        embedding_list.append(input_embedding)
                        label_list.append(int(output_prompt))
                        sample_list.append(input_prompt)
                    elif t_sne_type == 'rotten_tomatoes':
                        # To understand Enc(Text) among Good and Bad comments
                        embedding_list.append(input_embedding)
                        label_list.append(0 if "Good" in output_prompt else 1)
                    else:
                        assert t_sne_type is None, f"Unknown t_sne_type: {t_sne_type}"
                    
                    if debug == 3:
                        print(f"EMBEDDING: {embedding_list[-1]}")
                        print(f"LABLE: {label_list[-1]}")
                        if idx >= 7:
                            exit(0)
                else:
                    cos_sim = 0.0

                if split_tune_untune:
                    full_messages = get_messages(original_model_name, messages)
                    prompt = format_conversation(full_messages, tokenizer, plain=plain)
                    # Basename-based routing: --input_file often arrives as
                    # "datasets/hellaswag_test.jsonl", so a raw startswith()
                    # silently skipped the multiple-choice branch.
                    if os.path.basename(input_file or "").startswith("hellaswag"):
                        # BUG FIX: we were passing `max_new_tokens` (set to 8 by
                        # _mnt_for_dataset("hellaswag") in run_stp.sh) as the
                        # tokenizer `max_length`, which truncated every prompt
                        # to its first 8 tokens. The model then scored A/B/C/D
                        # against an empty question and collapsed to chance
                        # (~25%), regardless of how well training had gone.
                        # `relative_probability` needs the full *context* length
                        # (i.e. the model's max sequence length, same as at
                        # training time), not the generation budget.
                        rp_max_len = getattr(generation_config, "max_length", None) or 512
                        generated_response = relative_probability(model, tokenizer, prompt, max_length=rp_max_len,
                                                                  unmask_assistant_special_tokens=unmask_assistant_special_tokens)
                        if debug == 6:
                            print(f"<<< {prompt}")
                            print(f"=== {messages[2]['content']}")
                            print(f">>> {generated_response}")
                            exit(0)
                    else:
                        generated_response = generate_response(model, tokenizer, prompt, generation_config, max_new_tokens,
                                                               unmask_assistant_special_tokens)
                    # if generated_response == messages[2]["content"]:
                    # equal = (generated_response == messages[2]["content"])
                    # if startswith:
                    #     equal = generated_response.startswith(messages[2]["content"])
                    equal = eval(generated_response, messages, input_file, spider_path, startswith=False, debug=debug)
                    if _dataset_kind(input_file) == "synth":
                        synth_n += 1
                        gt_raw = messages[2]["content"]
                        rel_ok = synth_engineering_relaxed_match(generated_response, gt_raw)
                        if rel_ok:
                            synth_eng_hits += 1
                        if rel_ok and not equal:
                            synth_eng_relaxed_only += 1
                    if startswith:
                        is_startswith = eval(generated_response, messages, input_file, spider_path, startswith=True, debug=debug)
                        if is_startswith:
                            sim_list_startswith.append(cos_sim)
                    if debug == 2:
                        gen = repr(generated_response)
                        gt = repr(messages[2]["content"])
                        print(f"gt_vs_gen-{input_file}, {gt}, {gen}, {equal}")
                    if equal:
                        sim_list.append(cos_sim)
                    else:
                        sim_list_untune.append(cos_sim)
                else:
                    sim_list.append(cos_sim)
                
            except Exception as e:
                raise e

    if t_sne:
        data = {'embedding_list': embedding_list, 'label_list': label_list, 'sample_list': sample_list}
        with open ('tsne.pkl', 'wb') as f:
            pickle.dump(data, f)

    print(f"Success Rate: {model_name}, {len(sim_list) / (len(sim_list) + len(sim_list_untune))}", end="")
    if startswith:
        print(f", {len(sim_list_startswith) / (len(sim_list) + len(sim_list_untune))}")
    else:
        print()

    denom = len(sim_list) + len(sim_list_untune)
    if synth_n > 0 and denom == synth_n:
        er = synth_eng_hits / synth_n
        print(
            f"SYNTH_relaxed_engineering: {model_name}, {er:.6f}  ({synth_eng_hits}/{synth_n})",
            flush=True,
        )
        print(
            f"SYNTH_relaxed_only_vs_strict: {model_name}, {synth_eng_relaxed_only}/{synth_n} "
            "(count where relaxed holds but strict EM fails)",
            flush=True,
        )
    elif synth_n > 0:
        print(
            f"[synth-metrics] WARN: synth_n={synth_n} vs total scored={denom} — skip dual-summary",
            flush=True,
        )

    print(len(sim_list))
    if sim_list:
        print(sum(sim_list) / len(sim_list), np.std(sim_list))
        quantiles = np.quantile(sim_list, [0.1, 0.2, 0.5, 0.8, 0.9])
        print(quantiles)
    else:
        print("(sim_list is empty — skipping quantiles)")
    if split_tune_untune:
        print(len(sim_list_untune))
        if sim_list_untune:
            print(sum(sim_list_untune) / len(sim_list_untune), np.std(sim_list_untune))
            quantiles_fail = np.quantile(sim_list_untune, [0.1, 0.2, 0.5, 0.8, 0.9])
            print(quantiles_fail)
        else:
            print("(sim_list_untune is empty — skipping quantiles)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate assistant responses using a pretrained model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name or path (e.g., 'microsoft/DialoGPT-medium', './my-finetuned-model')")
    parser.add_argument("--original_model_name", type=str, required=True, 
                       help="Original model name.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Device map for model loading")
    
    # Data arguments
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with conversations")
    parser.add_argument("--output_file", type=str, help="Output JSONL file for generated responses")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to process")
    parser.add_argument("--no_skip_existing", action="store_true", help="Don't skip existing results in output file")
    
    # NEW: Train/Test split arguments
    parser.add_argument("--nosplit_data", action="store_true", 
                       help="Do not split input data into train and test sets before processing")
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Proportion of data to use for test set (default: 0.2)")
    parser.add_argument("--split_seed", type=int, default=42, 
                       help="Random seed for train/test split (default: 42)")
    parser.add_argument("--train_file", type=str, 
                       help="Output file for train set (auto-generated if not specified)")
    parser.add_argument("--test_file", type=str, 
                       help="Output file for test set (auto-generated if not specified)")
    parser.add_argument("--process_split", type=str, choices=['train', 'test', 'both'], default='test',
                       help="Which split to process for inference (default: both)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate. Use -1 to unset.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum total sequence length")
    parser.add_argument(
        "--eval_profile",
        type=str,
        default="fork",
        choices=list(VALID_EVAL_PROFILES),
        help=(
            "fork: honor --max_new_tokens/--max_length from CLI/shell (e.g. run_stp.sh "
            "per-dataset caps). llm_jepa_official: force reference defaults "
            f"(max_new_tokens={LLM_JEPA_OFFICIAL_MAX_NEW_TOKENS}, "
            f"max_length={LLM_JEPA_OFFICIAL_MAX_LENGTH}) for apples-to-apples comparison."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (1 = sampling)")
    parser.add_argument("--do_sample", action="store_true", help="Whether do sampling")

    # Similarity arguments
    parser.add_argument("--split_tune_untune", action="store_true", help="Whether to split result on tuned / untuned samples, where tuned means output is an exact match")
    parser.add_argument("--debug", type=int, default=0, help="Whether to print debug information")
    parser.add_argument("--start_index", type=int, default=1, help="The start index of messages to extract embedding, default to 1.")
    parser.add_argument("--embedding_layer", type=int, default=-1, help="Which layer to extract embedding, default to -1.")
    parser.add_argument("--embedding_pooling", type=str, choices=["last", "mean", "cls"], default="last", help="The pooling method for embedding")
    parser.add_argument("--similarity", action="store_true", help="Whether to compute similarity.")
    parser.add_argument("--t_sne", action="store_true", help="Whether to produce a t-SNE plot.")
    parser.add_argument("--t_sne_type", type=str, default=None, help="The t-SNE type, can be `in_n_out`, `paraphrase`, or`rotten_tomatoes`.")
    parser.add_argument("--startswith", action="store_true", help="Wither to report match if generated starts with ground-truth.")
    parser.add_argument("--plain", action="store_true", help="When set, do not apply chat format, and append `<|perception|>` to the prompt.")
    parser.add_argument(
        "--spider_path",
        type=str,
        default="spider_data/database",
        help="Path to spider databases.",
    )
    parser.add_argument("--unmask_assistant_special_tokens", action="store_true", help="When set, unmask assistant special tokens. Should match the training configuration.")

    
    args = parser.parse_args()
    if args.max_new_tokens == -1:
        args.max_new_tokens = None
    apply_eval_profile(args)

    # Validate arguments
    if not args.nosplit_data and not args.input_file and not (args.train_file and args.test_file):
        parser.error("When not using --nosplit_data, you must specify either --input_file or both --train_file and --test_file")
    
    if args.nosplit_data and not args.input_file:
        parser.error("You must specify --input_file when using --nosplit_data")
    
    print("=== Model Inference Script ===")
    print(f"Model: {args.model_name}")
    print(f"Input: {args.input_file}")
    
    if not args.nosplit_data:
        print(f"Split data: Yes (test_size={args.test_size}, seed={args.split_seed})")
        print(f"Process split: {args.process_split}")
    else:
        print(f"Output: {args.output_file}")
    
    print(f"Max examples: {args.max_examples or 'All'}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Do sample: {args.do_sample}")
    print(f"Don't skip existing: {args.no_skip_existing}")

    print(f"Split tune / untune: {args.split_tune_untune}")
    print(f"Start index: {args.start_index}")
    print(f"Embedding layer: {args.embedding_layer}")
    print(f"Embedding pooling: {args.embedding_pooling}")
    
    # Handle train/test splitting if requested
    if not args.nosplit_data:
        # Generate filenames if not provided
        base_name = os.path.splitext(args.input_file)[0]
        train_file = args.train_file or f"{base_name}_train.jsonl"
        test_file = args.test_file or f"{base_name}_test.jsonl"
        
        # Split the dataset
        train_file, test_file = split_dataset_and_save(
            args.input_file, train_file, test_file, 
            test_size=args.test_size, seed=args.split_seed
        )
        
        # Determine which files to process
        files_to_process = []
        if args.process_split in ['train', 'both']:
            output_train = args.output_file or f"{base_name}_train_responses.jsonl"
            files_to_process.append(('train', train_file, output_train))
        
        if args.process_split in ['test', 'both']:
            output_test = args.output_file or f"{base_name}_test_responses.jsonl"
            if args.process_split == 'both' and not args.output_file:
                output_test = f"{base_name}_test_responses.jsonl"
            files_to_process.append(('test', test_file, output_test))
    else:
        # Process single file
        files_to_process = [('full', args.input_file, args.output_file)]
    
    # Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.original_model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map
    )

    # Setup generation config
    print("\n2. Setting up generation configuration...")
    generation_config = GenerationConfig(
        model_name=args.model_name,
        do_sample=False, # args.do_sample,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        # temperature=args.temperature,
        # top_p=args.top_p,
        # top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Add model name to config for tracking
    # generation_config.model_name = args.model_name
    
    print(f"Generation config: {generation_config}")
    
    # Process each file
    print(f"\n3. Processing {len(files_to_process)} file(s)...")
    all_results = {}
    
    for split_name, input_file, output_file in files_to_process:
        print(f"\n--- Processing {split_name} set: {input_file} -> {output_file} ---")
        
        results = process_dataset(
            input_file=input_file,
            output_file=output_file,
            original_model_name=args.original_model_name,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            spider_path=args.spider_path,
            max_examples=args.max_examples,
            skip_existing=not args.no_skip_existing,
            split_tune_untune=args.split_tune_untune,
            layer=args.embedding_layer,
            pooling=args.embedding_pooling,
            start_index=args.start_index,
            debug=args.debug,
            similarity=args.similarity,
            startswith=args.startswith,
            max_new_tokens=args.max_new_tokens,
            t_sne=args.t_sne,
            t_sne_type=args.t_sne_type,
            plain=args.plain,
            model_name=args.model_name,
            unmask_assistant_special_tokens=args.unmask_assistant_special_tokens
        )
        
        all_results[split_name] = results
    
    print("\n🎉 Generation complete!")
    
    # Print summary
    if len(files_to_process) > 1:
        print("\nSummary:")
        for split_name in all_results:
            successful = len([r for r in all_results[split_name] if not r.get('failed', False)])
            failed = len([r for r in all_results[split_name] if r.get('failed', False)])
            print(f"  {split_name}: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
