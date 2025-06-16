import multiprocessing as mp
from functools import partial
from multiprocessing import Pool

from datasets import Dataset, load_dataset
from tqdm import tqdm
from deduplicate import fuzz_string_pair
from huggingface_hub import login
import os 
TOKEN = os.environ.get("HF_TOKEN", None)
if TOKEN is not None:
    login(token=TOKEN)
else:
    raise ValueError("HF_TOKEN is not set")
EVALUATION_DATASETS = {
    "HuggingFaceH4/MATH-500": {
        "eval_columns": ["problem"],
        "eval_splits": ["test"],
    },
    "Maxwell-Jia/AIME_2024": {
        "eval_columns": ["Problem"],
        "eval_splits": ["train"],
    },
    "AI-MO/aimo-validation-amc": {
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "yentinglin/aime_2025":{
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "tuenguyen/eval_math_olympiadbench":{
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "tuenguyen/eval_math_minerva_math":{
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "tuenguyen/eval_math_gaokao2023en":{
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "Intelligent-Internet/Math-Bench-final":{
        "eval_columns": ["problem_translation"],
        "eval_splits": ["train"],
    },
    "Intelligent-Internet/Physics-Entrance-Reform":{
        "eval_columns": ["reformulated_problem"],
        "eval_splits": ["train"],
    },
    "Intelligent-Internet/Chemistry-Entrance-Reform":{
        "eval_columns": ["reformulated_problem"],
        "eval_splits": ["train"],
    },
    "livecodebench/code_generation_lite": {
        "eval_columns": ["question_content"],
        "eval_splits": ["test"],
    },
    "Idavidrein/gpqa1": {
        "name": "Idavidrein/gpqa",
        "eval_columns": ["Question"],
        "eval_splits": ["train"],
        "eval_subset": "gpqa_diamond",
    },
    "Idavidrein/gpqa3": {
        "name": "Idavidrein/gpqa",
        "eval_columns": ["Question"],
        "eval_splits": ["train"],
        "eval_subset": "gpqa_extended",
    },
    "Idavidrein/gpqa4": {
        "name": "Idavidrein/gpqa",
        "eval_columns": ["Question"],
        "eval_splits": ["train"],
        "eval_subset": "gpqa_main",
    },
    "tuenguyen/eval_math_amc23":{
        "eval_columns": ["problem"],
        "eval_splits": ["train"],
    },
    "deepmind/code_contests":{
        "eval_columns": ["description"],
        "eval_splits": ["test"],
    },
    "google/IFEval":{
        "eval_columns": ["prompt"],
        "eval_splits": ["train"],
    },
    "cais/hle":{
        "eval_columns": ["question"],
        "eval_splits": ["test"],
    },
    "tuenguyen/eval_code_mpbb":{
        "eval_columns": ["text"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-GPQA_Medical_test":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-PubMedQA_test":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MedMCQA_validation":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-HumanityLastExam":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MedBullets_op5":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-NEJM":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MedXpertQA":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MedBullets_op4":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-Lancet":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MMLU-Pro_Medical_test":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/Medical-Eval-MedQA_USLME_test":{
        "eval_columns": ["question"],
        "eval_splits": ["train"],
    },
    "tuenguyen/MedXpertQA":{
        "eval_columns": ["question"],
        "eval_splits": ["test"],
    }
}

def decontaminate(dataset: Dataset, column="question", save_to_path=None, evals=EVALUATION_DATASETS, threshold=75.0) -> Dataset:
    """Remove rows from dataset that have similar strings in eval_datasets based on fuzzy matching."""
    n_processes = mp.cpu_count()

    # Get values from input dataset
    dataset_strings = [str(x) for x in dataset[column] if x is not None]
    indices_to_remove = set()
    #data_sources = list(dataset['data_source'])
    list_match = []
    match_dict = {}
    for eval_name, eval_info in list(evals.items())[::-1]:
        if 'name' in eval_info:
            eval_name = eval_info['name']
        eval_splits = eval_info["eval_splits"]
        eval_columns = eval_info["eval_columns"]
        eval_subset = eval_info.get("eval_subset", None)
        if eval_subset is not None:
            ds = load_dataset(eval_name, eval_subset, split=eval_splits, trust_remote_code=True)
        else:
            ds = load_dataset(eval_name, split=eval_splits, trust_remote_code=True)
        def format_problem(row):
            question = row['question']
            options = row['options']
            options_str = "\n".join([f"{key}. {value}" for key,value in options.items()])
            return f"{question}\n{options_str}"
        if "Medical-Eval" in eval_name:
            # print(ds)
            ds2 = ds[0].map(
                lambda x: {
                    eval_columns[0]:  format_problem(x)
                }
            )
            from datasets import concatenate_datasets
            ds = [concatenate_datasets([ds[0], ds2])]
        # for each split, column, and value
        eval_strings = [str(x) for split in ds for column in eval_columns for x in split[column] if x is not None]

        # Track indices to remove
        process_pair = partial(
            fuzz_string_pair,
            values2=eval_strings,
            similarity_threshold=threshold,
        )

        with Pool(n_processes) as pool:
            matches = list(
                tqdm(
                    pool.imap(process_pair, dataset_strings, chunksize=512),
                    total=len(dataset_strings),
                    desc=f"Decontaminating against {eval_name}_{eval_subset}",
                )
            )

        # Find indices where matches were found
        cnt = 0
        cnt_exact = 0
        
        for i, match_list in enumerate(matches):
            if any(score >= threshold for _, _, score in match_list):
                cnt += 1
                for src, dst, sc in match_list:
                    if sc >= threshold:
                        # if i not in indices_to_remove:
                        list_match.append(
                            dict(
                                    index=i,
                                    match_problem=src,
                                    score=sc,
                                    source=dst,
                                    #data_source=data_sources[i],
                                    eval_source=eval_name,
                            )
                        )
                            # break
                indices_to_remove.add(i)

                # print(match_list)
                # exit()
                for _,_, sc in match_list:
                    if sc == 100:
                        cnt_exact += 1
        set_list_match = set()
        list_match_unique = []
        for match in list_match:

            if match['match_problem'].strip() not in set_list_match and match['eval_source'] == eval_name:
                set_list_match.add(match['match_problem'].strip())
                list_match_unique.append(match)
        print(f"Removed {cnt} contaminated rows for {eval_name}")
        print(f"Exact match: {cnt_exact}")
        cnt = len(list_match_unique)
        cnt_exact = len([x for x in list_match_unique if x['score'] == 100])
        match_dict[f'{eval_name}_{eval_subset}'] = dict(
            cnt=cnt,
            cnt_exact=cnt_exact,
            # list_match=list_match,
        )
    keep_dataset = dataset.select(list(indices_to_remove)) 
    with open(save_to_path, "w") as f:
        for match in list_match_unique:
            import json
            f.write(json.dumps(match) + "\n")
    from datasets import Dataset 
    ds = Dataset.from_list(list_match_unique)
    from collections import Counter 
    # print(Counter(keep_dataset['data_source']))
    keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
    clean_dataset = dataset.select(keep_mask)

    print(f"Removed {len(indices_to_remove)} contaminated rows")
    print(f"Original size: {len(dataset)}, New size: {len(clean_dataset)}")

    return clean_dataset, ds, indices_to_remove, match_dict

def clean_problem(problem):
    problem = problem.replace("Return your final response as 'Final Answer: \\boxed{<answer>}', where <answer> is the number or mathematical expression of the solution.", "")
        # # problem = problem.split("Return your final response as")[0].strip()
    problem = problem.replace(
        "Please reason step by step, and put your final answer within \\boxed{}.",
        ""
    ).replace(
        "Return your final response as 'Final Answer: \\boxed{<answer>}', where <answer> is the number or mathematical expression of the solution.",
        ""
    ).replace(
        "Return your final response as \'Final Answer: \\boxed{<answer>}\', where <answer> is the number or mathematical expression of the solution.",
        ""
    ).replace("Return your final response within \\boxed{}", "").strip() 
    problem = problem.replace("Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output", "").strip()
    return problem

if __name__ == "__main__":
    import huggingface_hub
    from datasets import load_dataset, Dataset, load_from_disk
    all_ds = []
    from tqdm import tqdm
    import json
    # dataset = load_dataset("open-thoughts/OpenThoughts2-1M", cache_dir="/home/slurm/tuenv2/tuenv/hf_cache")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_save", type=str, required=True)
    parser.add_argument("--problem_column", type=str, default="problem")
    parser.add_argument("--threshold", type=float, default=90)
    args = parser.parse_args()
    try:
        dataset = load_from_disk(args.dataset_path)
    except:
        dataset = load_dataset(args.dataset_path)['train']
    if args.problem_column not in dataset.column_names:
        if "problem" in dataset.column_names:
            args.problem_column = "problem"
        elif "prompt" in dataset.column_names:
            args.problem_column = "prompt"
        elif "question" in dataset.column_names:
            args.problem_column = "question"
        else:
            raise ValueError(f"Problem column {args.problem_column} not found in dataset")
    dataset = dataset.map(
        lambda x: {
            "problem": clean_problem(x[args.problem_column])
        },
        num_proc=128
    )
    # if "problem" not in dataset.column_names:
    #     dataset = dataset.map(
    #         lambda x: {
    #             "problem": clean_problem(x["messages"][0]["content"])
    #         },
    #         num_proc=128
    #     )
    new_ds, ds, remove_indices, _= decontaminate(dataset, "problem", "decontaminated_medical_apollo_corpus.jsonl", threshold=args.threshold)
    # new_ds.push_to_hub("II-Vietnam/Trace-SFT-Filter-v0-Decontaminated-R1-V0-Decontaminated-R1-V1")
    # new_ds.save_to_disk("/home/slurm/tuenv2/open_i1_project/evaluation/medical_eval/m1/ds_light_decontaminated_round2")
    new_ds.save_to_disk(args.dataset_save)