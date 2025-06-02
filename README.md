# Medical Question Answering Evaluation Script

This script is designed to evaluate the performance of Large Language Models (LLMs) on a variety of medical question-answering (QA) datasets. It leverages the SGLang library for efficient model inference, processes model responses to extract answers, calculates several reward metrics, and saves detailed results along with aggregated performance metrics.

## Features

* **Multiple Dataset Support**: Evaluates models on a comprehensive suite of medical QA datasets from Hugging Face Hub, including:
    * `tuenguyen/Medical-Eval-MedMCQA_validation`
    * `tuenguyen/Medical-Eval-MedQA_USLME_test`
    * `tuenguyen/Medical-Eval-PubMedQA_test`
    * `tuenguyen/Medical-Eval-MMLU-Pro_Medical_test`
    * `tuenguyen/Medical-Eval-GPQA_Medical_test`
    * `tuenguyen/Medical-Eval-Lancet`
    * `tuenguyen/Medical-Eval-MedBullets_op4`
    * `tuenguyen/Medical-Eval-MedBullets_op5`
    * `tuenguyen/Medical-Eval-NEJM`
    * `tuenguyen/MedXpertQA`
    * `II-Vietnam/MMLU_test` (relevant medical subset if applicable, or general)
    * `II-Vietnam/AFRIMEDQA_test`
* **Efficient Inference**: Utilizes SGLang for running model inference, allowing for batch processing and optimized performance. (Requires an SGLang-compatible backend server).
* **Answer Extraction**: Implements multiple strategies for extracting answers from model responses:
    * Parsing content within LaTeX `\boxed{...}` commands.
    * A lenient version of `\boxed{...}` extraction.
    * Heuristics adapted from HuatuoGPT-o1 for matching multiple-choice options.
* **Reward Calculation**: Computes various reward scores:
    * Direct match of extracted boxed answers with the correct answer index.
    * Rewards based on the Huatuo answer matching logic.
    * A combined reward taking the best outcome from available methods.
* **Multiple Rollouts**: Supports running multiple evaluation rollouts to ensure robust and stable performance metrics.
* **Comprehensive Output**: Saves:
    * The configuration used for each run.
    * Detailed CSV files per dataset per rollout, containing prompts, responses, extracted answers, and rewards.
    * A summary JSON file (`metrics_summary.json`) with aggregated metrics per dataset (averaged over rollouts) and an overall average score.
    * Temporary JSON files for raw responses to allow resuming or debugging.

## Dependencies

* Python 3.8+
* `asyncio` (standard library)
* `pandas`
* `datasets` (from Hugging Face)
* `tqdm`
* `numpy`
* `sglang` (and its dependencies, which usually include an LLM serving library like vLLM or integration with Triton Inference Server). The script communicates with an SGLang server endpoint.
* `openai` (Python client library, used here to interface with the SGLang OpenAI-compatible API).
* `nest_asyncio`

You can install most Python dependencies using pip:
```bash
pip install pandas datasets tqdm numpy openai nest_asyncio sglang
```


## Setup

1. Clone Repository (if applicable) or download the script (medical.py) and ensure any utility scripts like sglang_util.py (if SGLangServerManager is custom) are in the correct path (e.g., a utils subdirectory).
2. Install Dependencies: As listed above.
3. SGLang Server: Ensure an SGLang server is running and accessible. This server should be configured to serve the desired LLM.

bash
```
python medical.py [OPTIONS]
Command-Line Arguments:
--system-prompt TEXT: (Optional) System prompt to guide the model's behavior.
--post-fix-prompt TEXT: Text appended after the question and options in the user prompt. (Default: "Please reason step by step, and put your final answer within \\boxed{}.")
--temperature FLOAT: Sampling temperature for generation. (Default: 0.6)
--top-p FLOAT: Nucleus sampling top_p value. (Default: 0.9)
--max-tokens INT: Maximum number of tokens to generate per response. (Default: 4096)
--output-dir TEXT: Directory where results and logs will be saved. (Default: "data")
--batch-size INT: Batch size for SGLang requests (effectively max_pending_requests). (Default: 64)
--num-rollouts INT: Number of times to run the evaluation for each dataset. (Default: 1)
--port INT: Starting port for the SGLang server API. (Default: 1234)
--override-temp: If set, overrides existing temporary response files instead of loading them.
```
Example Usage:

```
python medical.py \
    --system-prompt "You are a helpful medical assistant." \
    --output-dir results_run1 \
    --num-rollouts 1 \
    --batch-size 32
```


## Output Files

All outputs are saved in the directory specified by --output-dir.

* config.json: Stores the configuration parameters used for the run, including the actual model name fetched from the SGLang server.
* {output_dir}/{dataset_source_name}/responses_{rollout_number}.csv: A CSV file for each dataset and each rollout, containing:
    * problem_prompt: The full prompt sent to the model.
    * raw_response: The model's raw output.
    * extracted_answer_get_last_boxed: Answer extracted by get_last_boxed.
    * extracted_answer_extract_boxed_text: Answer extracted by extract_boxed_text_lenient.
    * reward_get_last_boxed: Reward for get_last_boxed (0 or 1).
    * reward_extract_boxed_text: Reward for extract_boxed_text_lenient (0 or 1).
    * reward_huatuo: Reward based on Huatuo matching (0 or 1).
    * final_reward_combined: Combined reward (max of the individual rewards).
    * correct_answer_index: The ground truth answer letter.
* {output_dir}/{dataset_source_name}/responses_{rollout_number}_tmp.json: (Optional) Temporary JSON dump of raw model responses for a rollout, useful for resuming.
* metrics_summary.json: A JSON file containing:
    * per_source: Metrics for each dataset, averaged over all rollouts (e.g., mean_final_reward_combined).
    * overall_average: Overall average metrics across specified datasets.
