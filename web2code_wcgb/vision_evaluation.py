import time
from typing import Callable

import requests
from PIL import Image

from .utils import DetailedScores, ImageEvaluationResult, MetricsResult, encode_image

# Model constants
MODEL_GPT4_TURBO = "gpt-4-turbo"
MODEL_GPT4_VISION = "gpt-4-vision-preview"
DEFAULT_MODEL = MODEL_GPT4_TURBO

# Prompts
system_prompt = """
You are an advanced AI model equipped with OCR and image processing capabilities, capable of analyzing visual elements in detail.
"""
user_prompt = """
Your task is to assess two webpage images and output a score between 0 and 10 for each of the following questions.
If the answer to a question is a definite YES, output a score of 10, signifying perfect similarity.
Conversely, a definite NO should yield a score of 0, indicating no similarity.
For answers that fall in between, assign a score accordingly, where a higher number indicates a greater degree of similarity. Only provide the numerical score for each question, without any additional text.
Example contexts are provided for clarity. Examples provides the idea, but you can output any number in 0-10
range accordingly. Only output a comma separated list containing 10 numbers. DO NOT give score of 10 for any category unless otherwise the two images are identical.

Layout Consistency (Score: 0-10): Does the placement of headers, footers, and sidebars match in both webpages? (e.g., A score of 10 for identical layouts, 5 for similar but not exact placements, and 0 for completely different layouts.)
Element Alignment (Score: 0-10): Are elements like images, buttons, and text boxes aligned similarly on both pages? (e.g., A score of 10 for perfectly aligned elements, 6 for slight misalignments, and 0 for major misalignments.)
Proportional Accuracy (Score: 0-10): Do the sizes and aspect ratios of images, buttons, and text boxes appear consistent across both pages? (e.g., A score of 10 for exact proportions, 4 for noticeable size differences, and 0 for drastic inconsistencies.)
Visual Harmony (Score: 0-10): Do both webpages exhibit a similar level of visual harmony and balance in their design? (e.g., A score of 10 for harmonious designs, 5 for some dissonance, and 0 for clashing designs.)

Color Scheme and Aesthetic Match (Score: 0-10): How closely do the color schemes of the two webpages align in terms of background and text colors? Evaluate the similarity in hues, saturation, and overall color aesthetics. (e.g., A score of 10 for perfectly matching color schemes, including identical hues and saturation levels, 6 for similar color palettes with minor variations, and 0 for starkly different color schemes that create entirely different visual impacts.)
Aesthetic Resemblance (Score: 0-10): Is the overall aesthetic appeal (modern, minimalistic, traditional, etc.) similar on both pages? (e.g., A score of 10 for identical aesthetics, 4 for somewhat similar but distinguishable styles, and 0 for completely different aesthetics.)

Font Characteristics and Consistency (Score: 0-10): Assess the degree of consistency in font attributes across both webpages. This includes not only the font type and size but also the nuances of font style (italic, bold) and weight (light, regular, bold). (e.g., A score of 10 for complete uniformity in font type, size, style, and weight across both pages, 5 for consistency in font type and size but variations in style or weight, and 0 for wide disparities in font type, size, style, or weight, leading to a distinctly different textual appearance.)
Textual Content Match (Score: 0-10): Do the words and sentences match between the two webpages? (e.g., A score of 10 for identical text, 5 for some similar paragraphs or sections, and 0 for completely different textual content.)
Numeric and Special Character Accuracy (Score: 0-10): Are numbers, dates, and special characters (like email addresses) consistent between the two pages? (e.g., A score of 10 for exact matches, 6 for minor discrepancies, and 0 for major differences.)

User Interface Consistency (Score: 0-10): Do the user interface elements (like menus, buttons, and forms) on both pages share a similar design language and appearance? (e.g., A score of 10 for identical UI elements, 6 for slight design variations, and 0 for completely different UI designs.)
"""


# Example definition, you can modify it as per your requirement
def fetch_api_response(
    gt_base64_image: str,
    pred_base64_image: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    max_attempts: int = 3,
    retry_delay: int = 5,
) -> str:
    """
    Fetch the API response with retries.

    Args:
        payload: The payload to send to the API.
        headers: The headers for the API request.
        max_attempts: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        The API response as a dictionary.

    Raises:
        Exception: If the API call fails after the maximum number of attempts.
    """

    # Set the headers
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{gt_base64_image}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{pred_base64_image}",
                        },
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
    }

    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.content.decode("utf-8")
        except Exception as e:
            print(f"Exception: {e}\n")
            attempt += 1
            time.sleep(retry_delay)
            continue
    raise Exception("Failed to fetch API response after retries.")


def generate_responses(
    pred_files: dict[str, Image.Image],
    gt_files: dict[str, Image.Image],
    cached_processed_data: dict[str, ImageEvaluationResult] | None = None,
    fetch_response_func: Callable[[str, str], str] = fetch_api_response,
) -> dict[str, ImageEvaluationResult]:
    """
    Generate evaluation responses for comparison between predicted and ground truth images.

    Args:
        pred_files: Dictionary mapping base_names to image objects of prediction images
        gt_files: Dictionary mapping base_names to image objects of ground truth images
        cached_processed_data: Dictionary of already processed data with image_id as keys
        fetch_response_func: Function to fetch API response, takes two str arguments and returns a str

    Returns:
        Dictionary containing the evaluation results with image_id as keys
    """
    if cached_processed_data is None:
        cached_processed_data = {}

    results = cached_processed_data.copy()

    # Process each pair of images
    for base_name, pred_image in pred_files.items():
        if base_name in results:
            print(f"Image {base_name} already processed.\n")
            continue

        # Check if we have corresponding ground truth
        if base_name not in gt_files:
            print(f"Ground truth image for {base_name} not found.")
            continue

        gt_image = gt_files[base_name]

        print(f"Processing image {base_name}...")

        # Encoding images to base64
        print(f"Encoding image {base_name}...")
        try:
            pred_base64_image = encode_image(pred_image)
            gt_base64_image = encode_image(gt_image)
        except Exception as e:
            print(f"Error encoding image {base_name}: {e}")
            continue

        try:
            content_str = fetch_response_func(gt_base64_image, pred_base64_image)

            # Add to results
            results[base_name] = {"image_id": base_name, "output": content_str}
            print(f"Image {base_name} processed successfully.\n")
        except Exception as e:
            print(f"Failed to process image {base_name}: {e}\n")

    return results


def evaluate_image_metrics(
    individual_results: dict[str, DetailedScores],
) -> MetricsResult:
    """
    Evaluate the image metrics based on the results.

    Args:
        results: Dictionary containing the results with image_id as keys

    Returns:
        Dictionary containing the evaluation metrics
    """
    total_images_processed = 0
    vis_struct = 0.0
    color_aesthetic = 0.0
    textual_content = 0.0
    user_interface = 0.0

    for output in individual_results.values():
        total_images_processed += 1
        vis_struct += (
            output["layout_consistency"]
            + output["element_alignment"]
            + output["proportional_accuracy"]
            + output["visual_harmony"]
        ) / 4
        color_aesthetic += (
            output["color_scheme_match"] + output["aesthetic_resemblance"]
        ) / 2
        textual_content += (
            output["font_consistency"]
            + output["textual_content_match"]
            + output["numeric_accuracy"]
        ) / 3
        user_interface += output["ui_consistency"]

    if total_images_processed == 0:
        return {
            "overall_similarity": 0,
            "visual_structure": 0,
            "color_aesthetic": 0,
            "textual_content": 0,
            "user_interface": 0,
        }

    vis_struct /= total_images_processed
    color_aesthetic /= total_images_processed
    textual_content /= total_images_processed
    user_interface /= total_images_processed
    overall_score = (
        vis_struct + color_aesthetic + textual_content + user_interface
    ) / 4

    return {
        "overall_similarity": overall_score,
        "visual_structure": vis_struct,
        "color_aesthetic": color_aesthetic,
        "textual_content": textual_content,
        "user_interface": user_interface,
    }


def get_individual_scores(
    results: dict[str, ImageEvaluationResult],
) -> dict[str, DetailedScores]:
    """
    Generate detailed scores for each sample based on the results.

    Args:
        results: Dictionary containing the results with image_id as keys

    Returns:
        Dictionary containing the detailed scores for each sample
    """
    individual_scores: dict[str, DetailedScores] = {}

    for image_id, data in results.items():
        output = data["output"]

        try:
            output_list = list(
                map(int, output.replace("\n", ",").strip(", ").split(","))
            )
            if len(output_list) != 10:
                print(f"Failed to process image: {image_id}")
                continue
            individual_scores[image_id] = DetailedScores(
                layout_consistency=output_list[0],
                element_alignment=output_list[1],
                proportional_accuracy=output_list[2],
                visual_harmony=output_list[3],
                color_scheme_match=output_list[4],
                aesthetic_resemblance=output_list[5],
                font_consistency=output_list[6],
                textual_content_match=output_list[7],
                numeric_accuracy=output_list[8],
                ui_consistency=output_list[9],
            )
        except Exception as e:
            print(f"Exception: {e}\n")
            continue

    return individual_scores
