from typing import Dict, Any, Optional, List, Tuple
import math_verify
from loguru import logger

def get_last_boxed(text: str):
    start_idx = text.rfind("\\boxed")
    if start_idx < 0:
        return None

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
    return text[start_idx : right_brace_idx + 1]


def verify(prediction: str, ground_truth: str): 
    extracted_prediction = get_last_boxed(prediction)
    if extracted_prediction is None:
        # print(f"Extracted prediction is None for {prediction} and {ground_truth}")
        return 0.0
    parsed_prediction = math_verify.parse(extracted_prediction)
    parsed_ground_truth = math_verify.parse(f"\\boxed{{{ground_truth}}}")
    try:
        return int(math_verify.verify(parsed_prediction, parsed_ground_truth))
    except Exception as e:
        logger.error(f"Error verifying {prediction} and {ground_truth}: {e}")
        return 0.0

