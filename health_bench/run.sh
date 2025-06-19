export A_API_KEY_41=
export LLM_BASE_ENDPOINT_41=
export DEPLOYMENT_NAME_41=gpt-4.1
export OPENAI_API_VERSION=2024-12-01-preview
export OPENAI_API_KEY=EMPTY

# bash run.sh save_hub/qwen3_32b_sft_1706_output_dir/health_bench 
# The final results will be save at save_hub/qwen3_32b_sft_1706_output_dir/health_bench/health_bench_final_all 

input_dir=$1
python3 infer.py --output-dir $input_dir # we generate all traces first b 
cd simple-evals
rm /tmp/healthbench_gpt*
python3 subset.py --input-dir-model-generated $input_dir/healthbench_generate.parquet --output-path $input_dir/all_results.jsonl
cd ..
python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1 --custom-input-path $input_dir/all_results.jsonl
mkdir -p $input_dir/health_bench_final_all
mv /tmp/healthbench_gpt* $input_dir/health_bench_final_all