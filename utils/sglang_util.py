import os
import subprocess
import time

import openai
import json
from .openai_server import OpenAIServerManager


def save_to_jsonl(data_list, filename):
    """
    Save a list of items to a JSONL file.
    Each item in the list will be written as a separate JSON line.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            # Convert each item to a JSON string and write with newline
            json_line = json.dumps(item)
            f.write(json_line + '\n')

def get_sglang_response(port, url="http://localhost", model_name="default"):
    """
    tries to get a response from the sglang server
    :return:
    """

    client = openai.Client(base_url=f"{url}:{port}/v1", api_key="EMPTY")

    # Text completion
    response = client.completions.create(
        model=model_name,
        prompt="The capital of France is",
        temperature=0,
        max_tokens=1,
    )
    print(response)


class SGLangServerManager(OpenAIServerManager):
    def launch_servers(
        self, model_name, start_port=1234, tp=1, max_time=160000,
    ):
        """
        Launches an sglang server on all available devices
        :param model_name:
        :param start_port: port to start on
        :param tp: tensor parallelism
        :param max_time: max time to wait
        :return:
        """
        count = 0
        print(f"launching sglang server on {self.url}:{start_port}, {model_name}")
        subprocesses = list()
        devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        ports = list()
        dp = len(devices) // tp
        for port in range(start_port, start_port + len(devices), tp):
            ports.append(start_port)
        # Since sglang handles dp internally, we just need to launch it once. We do keep multiple "ports" around
        # for compatibility with servers that don't have this option (cough vllm)
        
        # subprocesses.append(
        #     subprocess.Popen(
        #         f"python -m sglang.launch_server --model-path {model_name} --port {start_port} --tp {tp} --dp {dp} --log-level error",
        #         shell=True,
        #         preexec_fn=os.setsid,
        #     )
        # )
        # Ensure servers are running...
        ports_to_check = ports.copy()
        ports_working = list()
        while all([port not in ports_working for port in ports_to_check]):
            for port in ports_to_check:
                if port in ports_working:
                    continue
                try:
                    url=self.url
                    model_name = self.get_model_name()
                    get_sglang_response(port, url, model_name)
                    ports_working.append(port)
                except (openai.APITimeoutError, openai.APIConnectionError) as err:
                    print(f"Port {port} not ready yet")
                    time.sleep(10)
                    count += 1
                    if count * 10 > max_time:
                        raise err
        return ports, subprocesses