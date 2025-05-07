import os

from .vision_evaluation import generate_responces, read_jsonl_and_get_outputs


def gpt4_vision_evaluation(gt_output_dir, pred_output_dir, output_dir, api_key):
    """
    Evaluate the similarity between the ground truth and predicted webpage images using GPT4.

    Args:
    gt_output_dir: str, path to the ground truth webpage images
    pred_output_dir: str, path to the predicted webpage images

    Returns:
    None
    """

    # Create a JSONL file to store the outputs
    output_jsonl_path = os.path.join(output_dir, "gpt4_vision_evaluation_output.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate responses using GPT4
    generate_responces(pred_output_dir, gt_output_dir, output_jsonl_path, api_key)

    # Read the JSONL file and get the outputs
    read_jsonl_and_get_outputs(
        output_jsonl_path
    )  # Read the JSONL file and get the outputs
    read_jsonl_and_get_outputs(output_jsonl_path)
