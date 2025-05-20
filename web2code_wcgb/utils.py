import base64
import io
import json
import os
from pathlib import Path
from typing import TypedDict

from PIL import Image


class ImageEvaluationResult(TypedDict):
    image_id: str
    output: str


class DetailedScores(TypedDict):
    layout_consistency: int
    element_alignment: int
    proportional_accuracy: int
    visual_harmony: int
    color_scheme_match: int
    aesthetic_resemblance: int
    font_consistency: int
    textual_content_match: int
    numeric_accuracy: int
    ui_consistency: int


class MetricsResult(TypedDict):
    overall_similarity: float
    visual_structure: float
    color_aesthetic: float
    textual_content: float
    user_interface: float


def encode_image_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image(image: Image.Image) -> str:
    """
    Encode an image object to a base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_processed_data(output_jsonl_path: Path) -> dict[str, ImageEvaluationResult]:
    """
    Read processed data from JSONL file if it exists

    Args:
        output_jsonl_path: Path to the JSONL file

    Returns:
        Dictionary containing processed image data with image_id as keys
    """
    processed_data: dict[str, ImageEvaluationResult] = {}
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r") as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    if "image_id" in item:
                        processed_data[item["image_id"]] = item
    return processed_data


def save_results_to_jsonl(
    results: dict[str, ImageEvaluationResult], output_jsonl_path: Path
) -> None:
    """
    Save results to a JSONL file

    Args:
        results: Dictionary containing the results with image_id as keys
        output_jsonl_path: Path to the JSONL file
    """
    with open(output_jsonl_path, "w") as file:
        for result in results.values():
            file.write(json.dumps(result) + "\n")


def save_analysis_results(analysis_results: MetricsResult, output_file: Path) -> None:
    """
    Save analysis results to a file

    Args:
        analysis_results: Dictionary containing the analysis results
        output_file: Path to the output file
    """
    with open(output_file, "w") as file:
        file.write(
            f"Overall Similarity Score: {analysis_results['overall_similarity']:.4f}\n"
        )
        file.write(
            f"Visual_Structure_and_Alignment: {analysis_results['visual_structure']:.4f}\n"
        )
        file.write(
            f"Color and Aesthetic Design: {analysis_results['color_aesthetic']:.4f}\n"
        )
        file.write(
            f"Textual and Content Consistency: {analysis_results['textual_content']:.4f}\n"
        )
        file.write(
            f"User Interface and Interactivity: {analysis_results['user_interface']:.4f}\n"
        )


def normalize_input(
    input_data: str | list[str] | dict[str, str],
) -> dict[str, Image.Image]:
    """
    Normalize input to a dictionary mapping base_names to image objects

    Args:
        input_data: Either a directory path, a list of file paths, or a dictionary

    Returns:
        Dictionary mapping base_names to image objects
    """
    result = {}

    if isinstance(input_data, str):
        # Input is a directory
        if os.path.isdir(input_data):
            for file_name in os.listdir(input_data):
                if file_name.endswith((".png", ".jpg", ".jpeg")):
                    base_name = os.path.splitext(file_name)[0]
                    file_path = os.path.join(input_data, file_name)
                    with Image.open(file_path) as img:
                        result[base_name] = img.copy()

    elif isinstance(input_data, list):
        # Input is a list of file paths
        for file_path in input_data:
            if os.path.isfile(file_path) and file_path.endswith(
                (".png", ".jpg", ".jpeg")
            ):
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                with Image.open(file_path) as img:
                    result[base_name] = img.copy()

    elif isinstance(input_data, dict):
        # Input is already a dictionary
        for base_name, file_path in input_data.items():
            if os.path.isfile(file_path) and file_path.endswith(
                (".png", ".jpg", ".jpeg")
            ):
                with Image.open(file_path) as img:
                    result[base_name] = img.copy()

    return result
