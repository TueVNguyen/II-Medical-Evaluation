import os
import json
import glob
import logging
import argparse
import time
import random
import threading
from concurrent.futures import ProcessPoolExecutor
import openai
import multiprocessing



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("worker.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create a semaphore to limit concurrent API calls
# We use a multiprocessing semaphore since we're using ProcessPoolExecutor
api_semaphore = None  # Will be initialized in main()

def convert_timestamp(timestamp):
    """Convert a timestamp to a human-readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def call_api(prompt, model, base_url=None, max_retries=3, temperature=0.6, max_tokens=32768, top_p=0.95, top_k=20):
    """Call the OpenAI API with retries."""
    # Use the semaphore to limit concurrent API calls
    with api_semaphore:
        logger.info(f"Acquired API semaphore, making API call")
        client_kwargs = {"api_key": "EMPTY"}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = openai.Client(**client_kwargs)
        
        retries = 0
        retry_delay = 1
        messages=[{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and not isinstance(prompt, str):
            messages = prompt
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # top_k=top_k
                    extra_body={
                        # "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "response_information": {
                        "total_tokens": response.usage.total_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "is_complete": response.choices[0].finish_reason == "stop",
                        "trace_index": None,  # Will be set by the caller
                        "timestamp": convert_timestamp(time.time()),# convert to min/hour/day/month/year
                        "generate_config":{
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "top_k": top_k

                        }
                    }
                }
            except Exception as e:
                retries += 1
                logger.warning(f"API request failed (attempt {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    sleep_time = retry_delay * (2 ** (retries - 1)) * (0.5 + random.random())
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise
        
        raise Exception("Maximum retries exceeded")


def process_question(args):
    """Process a single question and trace."""
    question_file, question_hash, trace_index, output_dir, base_url, model = args
    
    response_path = os.path.join(output_dir, f"{question_hash}_{trace_index}.json")
    if os.path.exists(response_path):
        try:
            with open(response_path, 'r') as f:
                response_data = json.load(f)
            logger.info(f"Skipping {question_hash}, trace {trace_index} because it already exists")
            return True
        except Exception as e:
            logger.error(f"Error loading response for {question_hash}, trace {trace_index}: {e}")
            
    
    try:
        # Load the question
        question_text = question_file.get("question", "")
        
        # Call the API
        logger.info(f"Generating response for {question_hash}, trace {trace_index}")
        response_data = call_api(question_text, model, base_url)
        
        # Set the trace index
        response_data["response_information"]["trace_index"] = trace_index
        
        # Combine question and response data
        output_data = {**question_file, **response_data}
        
        # Save the response
        response_path = os.path.join(output_dir, f"{question_hash}_{trace_index}.json")
        with open(response_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully processed {question_hash}, trace {trace_index}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {question_hash}, trace {trace_index}: {e}", exc_info=True)
        return False
    
def get_model(base_url):
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    
    model_name = client.models.list()#.data[0].id
    if len(model_name.data) == 0:
        raise ValueError("No model found")
    model_name = model_name.data[0].id
    return model_name

def init_worker(semaphore):
    """Initialize worker process with the semaphore."""
    global api_semaphore
    api_semaphore = semaphore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LLM responses in parallel")
    parser.add_argument("--dataset-name", type=str, required=False, default="Intelligent-Internet/OpenAI-HealthBench-II-Medical-8B-GPT-4.1", 
                        help="The dataset name or path of HF dataset, which contains the question column and hash_value column and need_to_gen column")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to store response files")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1",
                        help="Base URL for the OpenAI API")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Model to use for generation")
    parser.add_argument("--num-workers", type=int, default=256, 
                        help="Number of worker processes")
    parser.add_argument("--num-traces", type=int, default=1, 
                        help="Number of traces to generate per question")
    parser.add_argument("--max-api-calls", type=int, default=256,
                        help="Maximum number of concurrent API calls")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    args.base_url="http://localhost:1235/v1"
    while True:
        try:
            args.model = get_model(args.base_url)
            break
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            import time
            time.sleep(10)
    from datasets import load_dataset, load_from_disk
    try:
        dataset_generate = load_dataset(args.dataset_name)['test']
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        dataset_generate = load_from_disk(args.dataset_name)

    work_items = []
    from tqdm import tqdm
    import hashlib
    for index, row in tqdm(enumerate(dataset_generate), total=len(dataset_generate)):

        question_file = {
            "question": row['prompt'],
            "other_information": {
                "index_in_ds": index,
            }
        }
        question_hash = hashlib.sha256(str(row['prompt']).encode('utf-8')).hexdigest()
        for trace_index in range(args.num_traces):
            work_items.append((
                    question_file, 
                    question_hash, 
                    trace_index, 
                    args.output_dir, 
                    args.base_url, 
                    args.model
                ))

    max_concurrent_calls = min(args.max_api_calls, args.num_workers)
    semaphore = multiprocessing.Semaphore(max_concurrent_calls)
    logger.info(f"Limiting to {max_concurrent_calls} concurrent API calls")
    import time
    #time.sleep(10) # sleep for 32 seconds to wait another nodes
    # Process questions in parallel
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(semaphore,)
    ) as executor:
    
        results = list(executor.map(process_question, work_items))
    
    # Report results
    successful = results.count(True)
    logger.info(f"Completed {successful} out of {len(work_items)} work items")

    # gather all the files in the output_dir
    output_files = glob.glob(os.path.join(args.output_dir, "*.json"))
    # sort the files by the question_hash
    output_files.sort(key=lambda x: x.split("_")[0])
    # merge the files
    merged_output = []
    for file in output_files:
        with open(file, 'r') as f:
            merged_output.append(json.load(f))
    import pandas as pd
    data_generate = pd.DataFrame(merged_output)
    print(data_generate.head())
    data_generate['index_in_ds'] = data_generate['other_information'].apply(lambda x: x['index_in_ds'])
    data_generate = data_generate[['index_in_ds', 'response', 'response_information']]
    data_generate = data_generate.groupby("index_in_ds").agg({"response": list, "response_information": list}).reset_index()
    data_generate = data_generate.sort_values(by="index_in_ds")
    print(data_generate.iloc[0].index_in_ds)
    index_to_response = {}
    index_to_response_information = {}
    for index, row in data_generate.iterrows():
        index_to_response[row['index_in_ds']] = row['response']
        index_to_response_information[row['index_in_ds']] = row['response_information']
    def map_index_to_response(row, index):
        if index not in index_to_response:
            return {
                "completion": "",
                "list_response": [""],
                "list_response_information": [""]
            }
        return {
            "completion": index_to_response[index][0].split("</think>")[-1].strip().replace("<think>", "").replace("</think>", ""),
            "list_response": index_to_response[index],
            "list_response_information": index_to_response_information[index]
        }
    dataset_generate = dataset_generate.map(map_index_to_response, with_indices=True)
    # print(dataset_generate[0]['prompt'])
    dataset_generate.to_parquet(os.path.join(args.output_dir, "healthbench_generate.parquet"))
    # print(data_generate.head())
    # data_generate.groupby("question_hash").agg({"response": "first"}).reset_index()
    # save the merged output

    