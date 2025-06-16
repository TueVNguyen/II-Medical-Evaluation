#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://arxiv.org/abs/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py

Usage:

python scripts/decontaminate.py \
    --dataset open-r1/verifiable-coding-problems-python \
    --split train \
    --ngram_size 8 \
    --problem_column problem \
    --cleanup
"""

import collections

from tqdm import tqdm


def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    ngram =  [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return [i for i in ngram if len(i.split()) == n] 

def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup

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
    ).replace("Return your final response within \\boxed{}", "").strip().replace(
        "Please reason step by step", "",
    ).strip()
    problem = problem.replace("Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output", "").strip()
    return problem

def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--config", type=str, default=None, help="Name of the dataset config to load.")
    parser.add_argument("--split", type=str, default="train", help="Split to check for contamination, defaults to `train`.")
    parser.add_argument("--ngram_size", type=int, default=10, help="Size of n-grams to build, defaults to 8.")
    parser.add_argument(
        "--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt)."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Whether to remove the contaminated rows before pushing the dataset.",
    )
    parser.add_argument(
        "--new_dataset_name",
        type=str,
        default=None,
        help="New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name."
    )
    args = parser.parse_args()

    from datasets import load_dataset, Dataset, load_from_disk

    # Load the dataset to check for contamination
    
    def format_problem(row):
        question = row['question']
        options = row['options']
        options_str = "\n".join([f"{key}. {value}" for key,value in options.items()])
        return f"{question}\n{options_str}"
    
    def get_v2_dataset(ds):
        ds = ds.map(lambda x: {"problem_formated": format_problem(x)}, num_proc=8)
        return ds

    eval_datasets = {
        # "ioi": (load_dataset("open-r1/ioi", split="train"), "statement"),
        # "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
        # "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
        # "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
        "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
        # "amc_validation": (load_dataset("AI-MO/aimo-validation-amc", split="train"), "problem"),
        # "math_olympiad_bench": (load_dataset("tuenguyen/eval_math_olympiadbench", split="train"), "problem"),
        # "minerva_math": (load_dataset("tuenguyen/eval_math_minerva_math", split="train"), "problem"),
        # "math_gaokao_en2023": (load_dataset("tuenguyen/eval_math_gaokao2023en", split="train"), "problem"),
        # "ii_math_bench": (load_dataset("Intelligent-Internet/Math-Bench-final", split="train"), "problem_translation"),
        # "physic_entrance_exam": (load_dataset("Intelligent-Internet/Physics-Entrance-Reform", split="train"), "reformulated_problem"),
        # "chemistry_entrance_exam": (load_dataset("Intelligent-Internet/Chemistry-Entrance-Reform", split="train"), "reformulated_problem"),
        # "eval_math_amc40": (load_dataset("tuenguyen/eval_math_amc23", split="train"), "problem"),
        # "code_contest": (
        #     load_dataset(
        #         "deepmind/code_contests", split="test", trust_remote_code=True
        #     ),
        #     "description",
        # ),
        # "IFEval": (
        #     load_dataset(
        #         "google/IFEval", split="train", trust_remote_code=True
        #     ),
        #     "prompt",
        # ),
        # "cais/hle": (
        #     load_dataset(
        #         "cais/hle", split="test", trust_remote_code=True
        #     ),
        #     "question",
        # ),
        # "eval_code_mpbb": (
        #     load_dataset(
        #         "tuenguyen/eval_code_mpbb", split="train", trust_remote_code=True
        #     ),
        #     "text",
        # ),
        
        # "lcb": (
        #     load_dataset(
        #         "livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True
        #     ),
        #     "question_content",
        # ),
        "tuenguyen/MedXpertQA": (
            load_dataset("tuenguyen/MedXpertQA", split="test"),
            "question"
        ),
        "tuenguyen/Medical-Eval-GPQA_Medical_test":(
            load_dataset("tuenguyen/Medical-Eval-GPQA_Medical_test", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-GPQA_Medical_test-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-GPQA_Medical_test", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-PubMedQA_test":(
            load_dataset("tuenguyen/Medical-Eval-PubMedQA_test", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-PubMedQA_test-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-PubMedQA_test", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MedMCQA_validation":(
            load_dataset("tuenguyen/Medical-Eval-MedMCQA_validation", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MedMCQA_validation-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MedMCQA_validation", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-HumanityLastExam":(
            load_dataset("tuenguyen/Medical-Eval-HumanityLastExam", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-HumanityLastExam-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-HumanityLastExam", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MedBullets_op5":(
            load_dataset("tuenguyen/Medical-Eval-MedBullets_op5", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MedBullets_op5-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MedBullets_op5", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-NEJM":(
            load_dataset("tuenguyen/Medical-Eval-NEJM", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-NEJM-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-NEJM", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MedXpertQA":(
            load_dataset("tuenguyen/Medical-Eval-MedXpertQA", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MedXpertQA-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MedXpertQA", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MedBullets_op4":(
            load_dataset("tuenguyen/Medical-Eval-MedBullets_op4", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MedBullets_op4-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MedBullets_op4", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-Lancet":(
            load_dataset("tuenguyen/Medical-Eval-Lancet", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-Lancet-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-Lancet", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MMLU-Pro_Medical_test":(
            load_dataset("tuenguyen/Medical-Eval-MMLU-Pro_Medical_test", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MMLU-Pro_Medical_test-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MMLU-Pro_Medical_test", split="train")),
            "problem_formated"
        ),
        "tuenguyen/Medical-Eval-MedQA_USLME_test":(
            load_dataset("tuenguyen/Medical-Eval-MedQA_USLME_test", split="train"),
            "question"
        ),
        "tuenguyen/Medical-Eval-MedQA_USLME_test-2":(
            get_v2_dataset(load_dataset("tuenguyen/Medical-Eval-MedQA_USLME_test", split="train")),
            "problem_formated"
        )
        
    }
    try:
        ds = load_dataset(args.dataset, name=args.config, split=args.split)
    except Exception as e:
        ds = load_from_disk(args.dataset)
    # print(ds)
    if args.problem_column not in ds.column_names:
        ds = ds.map(
            lambda x: {args.problem_column: x['messages'][-2]['content']}, # -1 is the last assistant message
            num_proc=128
        )
    if args.problem_column not in ds.column_names:
        if "problem" in ds.column_names:
            args.problem_column = "problem"
        elif "prompt" in ds.column_names:
            args.problem_column = "prompt"
        elif "question" in ds.column_names:
            args.problem_column = "question"
        else:
            raise ValueError(f"Problem column {args.problem_column} not found in dataset")
    ds = ds.map(
        lambda x: {args.problem_column: clean_problem(x[args.problem_column])},
        num_proc=128
    )
    ngram_lookups = {}
    for ds_name, (eval_dataset, problem_col) in eval_datasets.items():
        print(ds_name)
        ngram_lookups[ds_name] = build_ngram_lookup(eval_dataset[problem_col], ngram_size=args.ngram_size)

    
        # Update the ngram_lookup variable for each dataset
    def find_contaminated(row):
        # For each example we have to build the ngrams and check for all of them on each row
        for eval_name, ngram_lookup in ngram_lookups.items():
            ngrams = build_ngram_single(row[args.problem_column], ngram_size=args.ngram_size)
            row[f"contaminated_{eval_name}"] = any(set(ngram in ngram_lookup for ngram in ngrams))
        return row

    ds = ds.map(find_contaminated, num_proc=32)

    # Allow cleaning up via CLI args (removing the contaminated examples and dropping the columns)
    def cleanup(dataset: Dataset) -> Dataset:
        initial_size = len(dataset)
        contamination_cols = [col for col in dataset.column_names if col.startswith("contaminated_")]
        def filter_contaminated(x):
            for col in contamination_cols:
                if x[col]:
                    return False
            return True 
        # dataset = dataset.filter(filter_contaminated, num_proc=8)
        # for col in contamination_cols:
        #     if col.startswith("contaminated_"):
        #         size_prior = len(dataset)
        #         dataset = dataset.filter(lambda x: not x[col], num_proc=8)
        #         if len(dataset) < size_prior:
        #             print(f"Removed {size_prior - len(dataset)} samples from '{col.replace('contaminated_', '')}'")
        for col in contamination_cols:
            cols = dataset[col]
            total = [1 if x else 0 for x in cols]
            print(f"Total {col}: {sum(total)}")
            # cols = [x for x in cols if not x]

            # dataset = dataset.map(lambda x, col=col: {col: x[col]}, num_proc=8)
        dataset = dataset.filter(filter_contaminated, num_proc=8)
        dataset = dataset.remove_columns(contamination_cols)
        print(f"Initial size: {initial_size}, Final size: {len(dataset)}")
        return dataset
    #if "index_in_ds" not in ds.column_names:
    print("add index_in_ds column")
    if "index_in_ds" not in ds.column_names:
        ds = ds.add_column("index_in_ds", range(len(ds)))
    if args.cleanup:
        ds = cleanup(ds)

    new_ds_name = args.new_dataset_name or f"{args.dataset}_decontaminated"
    config_name = args.config if args.config is not None else "default"
    ds.save_to_disk(new_ds_name)
    # url = ds.push_to_hub(new_ds_name, config_name=config_name, split="train")
    print(f"Decontaminated dataset: {new_ds_name}")