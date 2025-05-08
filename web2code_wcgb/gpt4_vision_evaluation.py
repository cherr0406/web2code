from pathlib import Path

from .utils import (
    get_processed_data,
    normalize_input,
    save_analysis_results,
    save_results_to_jsonl,
)
from .vision_evaluation import (
    evaluate_image_metrics,
    fetch_api_response,
    generate_responses,
    get_individual_scores,
)


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
    output_dir_path = Path(output_dir)
    output_jsonl_path = output_dir_path / "gpt4_vision_evaluation_output.jsonl"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get the processed data from the JSONL file
    processed_data = get_processed_data(output_jsonl_path)

    # Generate responses using GPT4
    pred_files = normalize_input(pred_output_dir)
    gt_files = normalize_input(gt_output_dir)
    results = generate_responses(
        pred_files,
        gt_files,
        processed_data,
        fetch_response_func=lambda gt, pred: fetch_api_response(
            gt, pred, api_key=api_key
        ),
    )

    save_results_to_jsonl(results, output_jsonl_path)
    individual_results = get_individual_scores(results)
    analysis_results = evaluate_image_metrics(individual_results)
    analysis_output_file = output_dir / "vision_evaluation_summary.log"
    save_analysis_results(analysis_results, analysis_output_file)
