"""
Medical Question Answering Evaluation Script

This script evaluates Large Language Models (LLMs) on a variety of medical
question-answering datasets. It uses SGLang for model inference, processes
responses to extract answers, calculates reward metrics, and saves detailed
results and aggregated metrics.

Key functionalities:
- Loads and prepares multiple medical QA datasets from Hugging Face Hub.
- Generates model responses using SGLang.
- Extracts answers from responses using regex (targeting \\boxed{} expressions)
  and the Huatuo answer matching logic.
- Computes various reward scores based on extracted answers versus ground truth.
- Supports multiple evaluation rollouts for robustness.
- Saves configuration, detailed per-question results, and summary metrics.
"""
import asyncio
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import argparse
import json
import difflib
import re
import nest_asyncio
from typing import List, Dict, Any, Tuple, Optional, Union

nest_asyncio.apply() # Apply nest_asyncio for environments with an existing event loop.

# --- Constants ---
DEFAULT_SGLANG_MODEL_ALIAS: str = "default" # Alias for the model served by SGLang
DEFAULT_POST_FIX_PROMPT: str = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_TEMPERATURE: float = 0.6
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_TOKENS: int = 30768 # Adjusted from 30768 as it seemed excessively high for QA
DEFAULT_BATCH_SIZE: int = 128 # CCU Adjusted from 512, depending on GPU, 512 can be too large
DEFAULT_NUM_ROLLOUTS: int = 1
DEFAULT_PORT: int = 1234 # default port for SGLang server
DEFAULT_OUTPUT_DIR: str = "data"

STOP_SEQUENCES: List[str] = ["<eos>", "<end_of_turn>", "<|im_end|>", "<|eot_id|>"] # Common stop tokens

# --- Utility Functions ---

def str_similarity(str1: str, str2: str) -> float:
    """
    Calculates the similarity ratio between two strings.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        The similarity ratio (0.0 to 1.0).
    """
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def get_last_boxed(text: str) -> Optional[str]:
    """
    Extracts the content of the last LaTeX \\boxed{...} expression from the text.

    Args:
        text: The text to search within.

    Returns:
        The content within the last \\boxed{...}, or the first character of
        the text after "Answer:" if no \\boxed{...} is found but "Answer:" is present,
        otherwise None.
    """
    start_idx = text.rfind("\\boxed")
    if start_idx < 0:
        if "Answer:" in text:
            # Fallback for non-boxed answers starting with "Answer:"
            answer_part = text.split("Answer:")[-1].strip()
            if answer_part:
                # Attempt to get a concise answer, typically a letter for MCQs
                return answer_part.split(".")[0].strip()[:1]
        return None

    # Find the matching closing brace for the last \boxed command
    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    if not right_brace_idx:
        return None
    return text[start_idx : right_brace_idx + 1].replace("\\boxed", "").replace("\\text", "").replace("{", "").replace("}", "").strip()  


def find_most_similar_index(str_list: List[str], target_str: str) -> Optional[int]:
    """
    Finds the index of the most similar string in a list to a target string.

    Args:
        str_list: A list of strings to compare against.
        target_str: The target string.

    Returns:
        The index of the most similar string, or None if the list is empty.
    """
    if not str_list:
        return None

    most_similar_index = 0
    highest_similarity = 0.0

    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        if similarity >= highest_similarity:
            highest_similarity = similarity
            most_similar_index = i
    return most_similar_index

def huatuo_match_choice(text: str, option_letter: Dict[str, str]) -> Optional[str]:
    """
    Extracts the chosen option letter from a model's response using heuristics
    adapted from HuatuoGPT-o1's evaluation script.

    Args:
        text: The model's response text.
        option_letter: A dictionary mapping option letters (e.g., "A", "B") to
                     their textual descriptions.

    Returns:
        The matched option letter (e.g., "A", "B") or None if no clear match.
    """
    # For HuatuoGPT-o1 style output
    if "## Final Response\n\n" in text:
        text = text.split("## Final Response\n\n")[-1]

    # For strict prompt
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        # first_match_answer = matches[0].group(2)
        last_match_answer = matches[-1].group(2)
        return last_match_answer
    option_letter = {
        k: str(v) for k, v in option_letter.items()
    }
    # Non strict
    match_options = "ABCDEFGHIJKLMN"[: len(option_letter)]
    matches = list(
        re.finditer(
            r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["
            + match_options
            + r"])(\W|[\u4e00-\u9fff]|$)",
            text,
            re.S,
        )
    )
    if matches:
        # NOTE: We remove the trick from HuatuoGPT-o1, only consider the last match.
        # first_match_answer = matches[0].group(2)
        last_match_answer = matches[-1].group(2)
        return last_match_answer

    # Strictly find option text
    text = text.lower()
    option_letter_text_pairs = [
        (opt, text.rindex(str(option_letter[opt]).lower()))
        for opt in option_letter
        if str(option_letter[opt]).lower() in text
    ]
    if len(option_letter_text_pairs) > 0:
        last_match_answer = sorted(
            option_letter_text_pairs, key=lambda x: x[1], reverse=True
        )[0][0]

        # NOTE: We remove the trick from HuatuoGPT-o1, only consider the last match.
        # Try to match the first one
        # option_letter_text_pairs = [
        #     (opt, text.index(options[opt].lower()))
        #     for opt in options
        #     if options[opt].lower() in text
        # ]
        # first_match_answer = sorted(
        #     option_letter_text_pairs, key=lambda x: x[1], reverse=True
        # )[0][0]

        return last_match_answer

    # Fuzzy find option text
    else:
        option_letters = [x for x in option_letter]
        option_texts = [str(option_letter[x]).lower() for x in option_letter]
        most_similar_index = find_most_similar_index(option_texts, text.lower())
        return option_letters[most_similar_index]


def huato_reward(response, options, answer_idx):
    """
    Calculates a reward based on the Huatuo matching logic.

    Args:
        response: The model's generated response.
        options: A dictionary of option choices (e.g., {"A": "Text A", "B": "Text B"}).
        correct_answer_index: The correct option letter (e.g., "A").

    Returns:
        1 if the extracted answer matches the correct answer index, 0 otherwise.
    """
    if "</think>" in response:
        # we get the last think part (i.e answer)
        response = response.split("</think>")[-1]
    extracted_answer = huatuo_match_choice(response, options) # note we didn't use the answer_idx here to extract the answer
    return 1 if extracted_answer.lower() == answer_idx.lower() else 0

def extract_boxed_text_lenient(text: str, extract_mc_letter: bool = True) -> str:
    """
    Extracts content from the last LaTeX \\boxed{...} command in the text.
    More lenient version.

    Args:
        text: The text to extract boxed content from.
        extract_mc_letter: If True, tries to extract just the letter for
                           multiple-choice answers (e.g., A, B).

    Returns:
        The extracted content, or an empty string if no match.
    """
    # Regex to find \boxed{...}
    # Using re.S to make . match newlines as well
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)

    if not matches:
        return ""

    for match in matches[::-1]:
        if match == "":
            continue

        if extract_mc_letter:
            # Try to extract just the letter for multiple choice
            # Match patterns like: A, A., (A), A:, A), etc.
            mc_pattern = r'^([A-Z])[.:)\s]|^(([A-Z]))|^([A-Z])$'
            mc_match = re.search(mc_pattern, match.strip())
            if mc_match:
                # Return the first non-None group
                return next((g for g in mc_match.groups() if g is not None), "")

        return match
    if "Answer:" in match:
        return match.split("Answer:")[-1].strip().split(".")[0].strip()[:1]
    return ""

# --- Dataset Loading and Prompt Preparation ---

def load_and_prepare_datasets(max_sample: int = None) -> List[Dataset]:
    """
    Loads and prepares all specified medical QA datasets.

    Returns:
        A list of Hugging Face Datasets, each with a 'source' column.
    """
    print("Loading and preparing datasets...")
    list_of_ds_names = [
        "tuenguyen/Medical-Eval-MedMCQA_validation",
        "tuenguyen/Medical-Eval-MedQA_USLME_test",
        "tuenguyen/Medical-Eval-PubMedQA_test",
        "tuenguyen/Medical-Eval-MMLU-Pro_Medical_test",
        "tuenguyen/Medical-Eval-GPQA_Medical_test",
        "tuenguyen/Medical-Eval-Lancet",
        "tuenguyen/Medical-Eval-MedBullets_op4",
        "tuenguyen/Medical-Eval-MedBullets_op5",
        "tuenguyen/Medical-Eval-NEJM"
    ]
    
    all_datasets = []
    for ds_name in tqdm(list_of_ds_names, desc="Loading benchmark datasets"):
        try:
            dataset = load_dataset(ds_name)['train']
            dataset = dataset.map(lambda x: {"source": ds_name})
            if max_sample:
                dataset = dataset.select(range(max_sample))
            all_datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Could not load or process {ds_name}. Error: {e}")

    try:
        medexpert_qa = load_dataset("tuenguyen/MedXpertQA")['test']
        medexpert_qa = medexpert_qa.map(
            lambda x: {
                "question": x['question'].split("Answer Choices:")[0].strip(),
                "options": x['options'], # Assuming options is already a dict like {"A": "text", ...}
                "answer": x['options'][x['label']],
                "answer_idx": x['label'],
                "source": "MedXpertQA"
            }
        )
        if max_sample:
            medexpert_qa = medexpert_qa.select(range(max_sample))
        all_datasets.append(medexpert_qa)
    except Exception as e:
        print(f"Warning: Could not load or process MedXpertQA. Error: {e}")

    # MMLU and AFRIMEDQA might have different structures
    # MMLU and AFRIMEDQA will take too long to run
    # try:
    #     data_mmlu = load_dataset("II-Vietnam/MMLU_test")['train']
    #     # Assuming MMLU needs 'source', 'answer_idx' if not present
    #     # This requires knowing MMLU's structure. For now, just add source.
    #     data_mmlu = data_mmlu.map(lambda x: {"source": "II-Vietnam/MMLU_test"})
    #     all_datasets.append(data_mmlu)
    # except Exception as e:
    #     print(f"Warning: Could not load or process II-Vietnam/MMLU_test. Error: {e}")
    
    # try:
    #     data_afrimedqa = load_dataset("II-Vietnam/AFRIMEDQA_test")['train']
    #     data_afrimedqa = data_afrimedqa.map(lambda x: {"source": "II-Vietnam/AFRIMEDQA_test"})
    #     all_datasets.append(data_afrimedqa)
    # except Exception as e:
    #     print(f"Warning: Could not load or process II-Vietnam/AFRIMEDQA_test. Error: {e}")


    # Reverse order as in original script, though unclear if necessary
    return all_datasets[::-1]


def convert_options_to_str(options_dict: Dict[str, str]) -> str:
    """
    Converts a dictionary of options to a formatted string.
    Example: {"A": "Apple", "B": "Banana"} -> "A. Apple\nB. Banana"
    """
    if not isinstance(options_dict, dict): # Handle cases where options might be missing or malformed
        return ""
    sorted_keys = sorted(options_dict.keys())
    return "\n".join([f"{k}. {options_dict[k]}" for k in sorted_keys])


def prepare_prompts(dataset: Dataset, system_prompt: Optional[str], post_fix_prompt: str) -> List[List[Dict[str, str]]]:
    """
    Prepares a list of prompts in chat format for the SGLang server.

    Args:
        dataset: The input Hugging Face Dataset.
        system_prompt: An optional system prompt string.
        post_fix_prompt: A string to append to each user prompt (e.g., instructions).

    Returns:
        A list of prompts, where each prompt is a list of message dictionaries.
    """
    prompts = []
    for row in dataset:
        question = row.get('question', '')
        options_dict = row.get('options', {}) # options should be a dict like {"A": "text", ...}

        # Ensure options_dict is a dictionary before processing
        if not isinstance(options_dict, dict):
            options_str = "" # Or handle as an error/skip
        else:
            options_str = convert_options_to_str(options_dict)

        # AFRIMEDQA has a specific format, doesn't include options in the prompt directly.
        if row.get('source') == "II-Vietnam/AFRIMEDQA_test" or row.get('source') == "AFRIMEDQA":
             user_content = question
        else:
            user_content = f"{question}\n{options_str}\n\n{post_fix_prompt}"
        
        messages = [{"role": "user", "content": user_content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        prompts.append(messages)
    return prompts

# --- Evaluation Logic ---
# Note: SGLangServerManager is expected to be imported from utils.sglang_util
# Ensure sglang_util.py is in the Python path or in the same directory.
try:
    from utils.sglang_util import SGLangServerManager
except ImportError:
    # Fallback if utils.sglang_util is not directly in PYTHONPATH, try local import
    # This assumes sglang_util.py might be in a 'utils' subdirectory relative to medical.py
    try:
        from .utils.sglang_util import SGLangServerManager
    except ImportError:
        print("Error: SGLangServerManager could not be imported. Ensure sglang_util.py is accessible.")
        SGLangServerManager = None # Make it None to cause errors if used without import


async def process_dataset_rollout(
    dataset: Dataset,
    server_handler: Any, # SGLangServerManager instance
    config: Dict[str, Any],
    rollout_num: int,
    output_dir_source: str
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]], Optional[List[int]], Optional[List[int]], Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """
    Processes a single dataset for one rollout: generates responses, extracts answers, calculates rewards.
    """
    source_name = dataset['source'][0].split("/")[-1] if dataset['source'] else "unknown_source"
    print(f"Processing {source_name}, Rollout {rollout_num + 1}/{config['num_rollouts']}...")

    prompts = prepare_prompts(dataset, config["system_prompt"], config["post_fix_prompt"])
    
    temp_responses_file = f"{output_dir_source}/responses_{rollout_num}_tmp.json"
    
    if os.path.exists(temp_responses_file) and not config.get("override_temp", False):
        print(f"Loading temporary responses from {temp_responses_file}")
        with open(temp_responses_file, "r") as f:
            raw_responses = json.load(f)
    else:
        if not server_handler:
            print("Error: SGLangServerManager not available. Skipping response generation.")
            return None, None, None, None, None, None, None, None
            
        print(f"Generating {len(prompts)} responses for {source_name}...")
        # Ensure prompts list is not empty
        if not prompts:
            print(f"Warning: No prompts generated for {source_name}. Skipping.")
            return None, None, None, None, None, None, None, None

        api_responses = await server_handler.get_chat_responses(
            prompts,
            n=1,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            stop=STOP_SEQUENCES
        )
        # Extract the actual response string
        raw_responses = [str(r[-1]["responses"][0]) if r and r[-1].get("responses") else "<ERROR_NO_RESPONSE>" for r in api_responses]

        try:
            with open(temp_responses_file, "w") as f:
                json.dump(raw_responses, f)
        except IOError as e:
            print(f"Warning: Could not write temporary responses to {temp_responses_file}. Error: {e}")

    # Ensure all required keys exist in the dataset rows
    # Default to "A" or empty string if 'answer_idx' or 'options' are missing.
    # This is crucial for robust processing of diverse datasets.
    answer_indices = [row['answer_idx'] for row in dataset]
    options_list = [row['options'] for row in dataset]

    extracted_answers_boxed = [get_last_boxed(r) for r in raw_responses]
    extracted_answers_lenient_boxed = [extract_boxed_text_lenient(r) for r in raw_responses]

    # Rewards based on get_last_boxed
    rewards_boxed = [
        1 if str(ext_ans).lower() == str(corr_idx).lower() else 0
        for ext_ans, corr_idx in zip(extracted_answers_boxed, answer_indices)
    ]
    # Rewards based on extract_boxed_text_lenient
    rewards_lenient_boxed = [
        1 if str(ext_ans).lower() == str(corr_idx).lower() else 0
        for ext_ans, corr_idx in zip(extracted_answers_lenient_boxed, answer_indices)
    ]

    rewards_huatuo_list = [
        huato_reward(resp, opts, corr_idx)
        for resp, opts, corr_idx in zip(raw_responses, options_list, answer_indices)
    ]

    # Combine rewards: take the max, preferring any successful extraction method
    final_rewards_combined = [max(rb, rh, rlb) for rb, rh, rlb in zip(rewards_boxed, rewards_huatuo_list, rewards_lenient_boxed)]


    results_df = pd.DataFrame({
        "problem_prompt": [p[0]['content'] if p else "" for p in prompts], # Assuming single turn user prompt
        "raw_response": raw_responses,
        "extracted_answer_get_last_boxed": extracted_answers_boxed,
        "extracted_answer_extract_boxed_text": extracted_answers_lenient_boxed,
        "reward_get_last_boxed": rewards_boxed,
        "reward_extract_boxed_text": rewards_lenient_boxed,
        "reward_huatuo": rewards_huatuo_list,
        "final_reward_combined": final_rewards_combined,
        "correct_answer_index": answer_indices
    })
    
    try:
        results_df.to_csv(f"{output_dir_source}/responses_{rollout_num}.csv", index=False)
    except IOError as e:
        print(f"Warning: Could not write CSV results to {output_dir_source}/responses_{rollout_num}.csv. Error: {e}")

    return (results_df, raw_responses, extracted_answers_boxed, rewards_boxed, 
            rewards_huatuo_list, final_rewards_combined, 
            extracted_answers_lenient_boxed, rewards_lenient_boxed)


async def run_evaluation(config: Dict[str, Any]):
    """
    Main function to run the LLM evaluation pipeline.
    """
    if SGLangServerManager is None:
        print("SGLangServerManager not imported. Cannot proceed with evaluation.")
        return

    all_datasets = load_and_prepare_datasets(config.get("max_sample", None))
    if not all_datasets:
        print("No datasets loaded. Exiting.")
        return

    # This dictionary will store results across rollouts for final aggregation
    # Structure: { "source_name": { "metric_name": [ [rollout1_val1, rollout1_val2,...], [rollout2_val1, ...], ... ] } }
    # Simpler: store lists of mean scores per rollout: { "source_name": { "metric_name": [mean_rollout1, mean_rollout2,...] } }
    aggregated_metrics_per_source: Dict[str, Dict[str, List[float]]] = {}


    # `all_rollout_raw_data` stores detailed data if needed for multi-rollout analysis later.
    # Structure: { "source_name": { "responses": [[r1_q1, r1_q2,...], [r2_q1, r2_q2,...]], ... } }
    # For simplicity, the current script saves per-rollout CSVs and then calculates means from those.
    # If more complex cross-rollout aggregation is needed later, this structure can be populated.
    # all_rollout_raw_data: Dict[str, Dict[str, List[List[Any]]]] = {}


    os.makedirs(config["output_dir"], exist_ok=True)
    with open(f"{config['output_dir']}/config.json", "w") as f:
        # Store a copy of the config used for this run
        # Add model name if obtained from server later
        json.dump(config, f, indent=4)

    # Initialize SGLang server manager
    # The model name for SGLang is "default" here, actual model name is fetched from server
    with SGLangServerManager(
        DEFAULT_SGLANG_MODEL_ALIAS,
        tp=8, # Tensor Parallelism, make this configurable if needed
        max_pending_requests=config["batch_size"],
        start_port=config["port"]
    ) as server_handler:
        
        try:
            actual_model_name = server_handler.get_model_name()
            print(f"Connected to SGLang server. Model: {actual_model_name}")
            config["actual_model_name"] = actual_model_name
            with open(f"{config['output_dir']}/config.json", "w") as f: # Update config with actual model name
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not get model name from server. Error: {e}")
            config["actual_model_name"] = "Unknown (Failed to fetch)"


        current_run_metrics: Dict[str, float] = {} # Metrics for the current overall run (averaged over sources)

        for rollout_num in range(config["num_rollouts"]):
            print(f"\n--- Starting Rollout {rollout_num + 1}/{config['num_rollouts']} ---")
            
            # Metrics for this specific rollout, to be printed at the end of the rollout
            rollout_summary_metrics: Dict[str, List[float]] = {
                "final_reward_combined": [],
                "reward_get_last_boxed": [],
                "reward_extract_boxed_text": [],
                "reward_huatuo": [],
            }

            for dataset in tqdm(all_datasets, desc=f"Rollout {rollout_num + 1} Datasets"):
                source_name = dataset['source'][0].split("/")[-1]
                output_dir_source = f"{config['output_dir']}/{source_name}"
                os.makedirs(output_dir_source, exist_ok=True)

                (   df_results, raw_responses_list, extracted_answers_boxed_list, 
                    rewards_boxed_list, rewards_huatuo_list, final_rewards_combined_list,
                    extracted_answers_lenient_list, rewards_lenient_list
                ) = await process_dataset_rollout(dataset, server_handler, config, rollout_num, output_dir_source)

                if df_results is None: # Skip if processing failed
                    continue

                # Store metrics for this source and this rollout
                if source_name not in aggregated_metrics_per_source:
                    aggregated_metrics_per_source[source_name] = {
                        "final_reward_combined": [],
                        "reward_get_last_boxed": [],
                        "reward_extract_boxed_text": [],
                        "reward_huatuo": []
                    }
                
                # Append mean scores for the current rollout to the list for this source
                aggregated_metrics_per_source[source_name]["final_reward_combined"].append(df_results["final_reward_combined"].mean())
                aggregated_metrics_per_source[source_name]["reward_get_last_boxed"].append(df_results["reward_get_last_boxed"].mean())
                aggregated_metrics_per_source[source_name]["reward_extract_boxed_text"].append(df_results["reward_extract_boxed_text"].mean())
                aggregated_metrics_per_source[source_name]["reward_huatuo"].append(df_results["reward_huatuo"].mean())
                
                rollout_summary_metrics["final_reward_combined"].append(df_results["final_reward_combined"].mean())
                rollout_summary_metrics["reward_get_last_boxed"].append(df_results["reward_get_last_boxed"].mean())
                rollout_summary_metrics["reward_extract_boxed_text"].append(df_results["reward_extract_boxed_text"].mean())
                rollout_summary_metrics["reward_huatuo"].append(df_results["reward_huatuo"].mean())

            # Print summary for the completed rollout
            print(f"\n--- Summary for Rollout {rollout_num + 1} (Excluding MMLU & AFRIMEDQA from average) ---")
            for metric_name, scores_list in rollout_summary_metrics.items():
                if scores_list: # Check if list is not empty
                    mean_score = sum(scores_list) / len(scores_list)
                    print(f"Average {metric_name}: {mean_score:.4f}")
                else:
                    print(f"Average {metric_name}: N/A (no applicable datasets processed)")
            print("----------------------------------------------------------------")


        # After all rollouts, calculate final mean metrics across rollouts for each source
        final_metrics_output: Dict[str, Any] = {"per_source": {}}
        for source_name, metrics_dict in aggregated_metrics_per_source.items():
            final_metrics_output["per_source"][source_name] = {}
            for metric_name, rollout_scores in metrics_dict.items():
                if rollout_scores:
                    final_metrics_output["per_source"][source_name][f"mean_{metric_name}"] = sum(rollout_scores) / len(rollout_scores)
                    # Optionally, also store std_dev/scores per rollout or [individual rollout scores] if needed
                    # final_metrics_output["per_source"][source_name][f"{metric_name}_rollouts"] = rollout_scores
                else:
                    final_metrics_output["per_source"][source_name][f"mean_{metric_name}"] = "N/A"


        # Calculate overall average metrics (excluding MMLU, AFRIMEDQA)
        overall_avg_metrics: Dict[str, List[float]] = {
            "final_reward_combined": [], "reward_get_last_boxed": [],
            "reward_extract_boxed_text": [], "reward_huatuo": []
        }
        for source_name, source_metrics in final_metrics_output["per_source"].items():
            # Normalize source name for checking exclusion
            norm_source_name = source_name.replace("_test","").replace("_validation","")
                            
            if norm_source_name not in ["MMLU", "AFRIMEDQA", "Medical-Eval-MMLU-Pro_Medical", "AFRIMEDQA"]: # Add variations
                for metric_key in overall_avg_metrics.keys():
                    mean_metric_val = source_metrics.get(f"mean_{metric_key}")
                    if isinstance(mean_metric_val, float): # Check if it's a valid float
                         overall_avg_metrics[metric_key].append(mean_metric_val)
        
        final_metrics_output["overall_average"] = {}
        print("\n--- Overall Evaluation Summary (Averaged across rollouts and specified sources) ---")
        for metric_name, scores in overall_avg_metrics.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                final_metrics_output["overall_average"][f"mean_{metric_name}"] = avg_score
                print(f"Overall Average {metric_name}: {avg_score:.4f}")
            else:
                final_metrics_output["overall_average"][f"mean_{metric_name}"] = "N/A"
                print(f"Overall Average {metric_name}: N/A")
        
        try:
            with open(f"{config['output_dir']}/metrics_summary.json", "w") as f:
                json.dump(final_metrics_output, f, indent=4)
            print(f"\nFinal aggregated metrics saved to {config['output_dir']}/metrics_summary.json")
        except IOError as e:
            print(f"Warning: Could not write final metrics summary. Error: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs on Medical QA Datasets using SGLang.")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the model.")
    parser.add_argument("--post-fix-prompt", type=str, default=DEFAULT_POST_FIX_PROMPT, help="Text appended after question and options.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling top_p.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum tokens to generate.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save results.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for SGLang requests (max_pending_requests).")
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS, help="Number of evaluation rollouts.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Starting port for SGLang server communication.")
    parser.add_argument("--override-temp", action='store_true', help="If set, override existing temporary response files.")
    parser.add_argument("--max-sample", type=int, default=None, help="Maximum number of samples to process.")
    args = parser.parse_args()

    # Prepare configuration dictionary
    run_config = {
        "system_prompt": args.system_prompt,
        "post_fix_prompt": args.post_fix_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_rollouts": args.num_rollouts,
        "port": args.port,
        "override_temp": args.override_temp,
        "max_sample": args.max_sample,
        # Note: Actual model name will be added once connected to server
    }

    print("Starting Medical QA Evaluation with configuration:")
    for key, value in run_config.items():
        print(f"  {key}: {value}")
    
    try:
        asyncio.run(run_evaluation(run_config))
    except Exception as e:
        print(f"An error occurred during the evaluation run: {e}")
        import traceback
        traceback.print_exc()