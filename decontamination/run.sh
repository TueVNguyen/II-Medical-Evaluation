# dataset_path='II-Vietnam/Qwen3-PubmedQA'
set -x
conda env list
eval "$(conda shell.bash hook)"
conda activate 360-llama-factory-v2
export HF_TOKEN=..
dataset_path="hoanganhpham/ChatDoctor-HealthCareMagic-Output-Improved-GPT4.1"

for dataset_path in  "/home/slurm/tuenv2/tuenv/exp_medical/datahub_v3/final_ds_05_06_combined_HA"
do
    python3 ngram_decontaminate.py --dataset $dataset_path \
        --problem_column question \
        --ngram_size 10 \
        --cleanup

    dataset_path_after_ngram=$dataset_path #'_decontaminated'
    python3 decontain.py --dataset_path $dataset_path_after_ngram \
        --dataset_save $dataset_path_after_ngram-fuzzy-decontaminated \
        --problem_column question \
        --threshold 90
done
