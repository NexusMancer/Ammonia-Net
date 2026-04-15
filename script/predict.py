import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config.config import PredictConfig, get_predict_config
from engine.inference import AmmoniaNetInferencer


def parse_args(config: PredictConfig) -> argparse.Namespace:
    """Parse command-line arguments for batch/single-image prediction.
    
    Creates argument parser with options for model loading, visualization, output paths,
    and segmentation/grading class names. Defaults are loaded from config object.
    
    Args:
        config: PredictConfig object containing default values and settings.
        
    Returns:
        Namespace object with parsed command-line arguments.
        
    Arguments documented:
        - input_path: Required path to image file or directory
        - output: Optional output path (file for single image, directory for batch)
        - model_path: Path to trained model weights
        - mix_type: Visualization mode (0=blend, 1=mask only, 2=foreground only)
        - cpu: Force CPU inference instead of GPU
        - count: Enable pixel statistics printing
        - draw_grade_label: Overlay grading result on output image
        - font_path/size: Font settings for grade label rendering
    """
    parser = argparse.ArgumentParser(description="Run AmmoniaNet on image files only.")
    parser.add_argument(
        "input_path",
        help="Path to one image file or a directory of images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path for single-image inference, or output directory for batch inference.",
    )
    parser.add_argument(
        "--model-path",
        default=config.inference.model_path,
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--mix-type",
        type=int,
        choices=(0, 1, 2),
        default=config.inference.mix_type,
        help="0: blend image and mask, 1: mask only, 2: foreground only.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print segmentation pixel statistics.",
    )
    parser.add_argument(
        "--draw-grade-label",
        action="store_true",
        help="Draw the grading label on the saved prediction image.",
    )
    parser.add_argument(
        "--font-path",
        default=config.font_path,
        help="Optional font file used when drawing the grading label.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=config.font_size,
        help="Font size used when drawing the grading label.",
    )
    parser.add_argument(
        "--suffix",
        default=config.output_suffix,
        help="Suffix appended to saved filenames when output is not an explicit file path.",
    )
    parser.add_argument(
        "--seg-classes",
        nargs="+",
        default=list(config.seg_classes),
        help="Segmentation class names used by --count.",
    )
    parser.add_argument(
        "--grade-classes",
        nargs="+",
        default=list(config.grade_classes),
        help="Grading labels mapped from the classification head output.",
    )
    return parser.parse_args()


def collect_image_paths(input_path: Path, config: PredictConfig) -> List[Path]:
    """Collect image file paths from input location (single file or directory).
    
    Handles two input modes:
    1. Single file: validates and returns the single image path
    2. Directory: scans directory and collects all supported image files
    
    Args:
        input_path: Path object pointing to image file or directory.
        config: PredictConfig containing supported image file extensions.
        
    Returns:
        List of Path objects for valid image files found.
        
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If input file has unsupported extension or directory has no images.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in config.input_image_suffixes:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    image_paths = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in config.input_image_suffixes
    )
    if not image_paths:
        raise ValueError(f"No supported images found in directory: {input_path}")
    return image_paths


def resolve_grade_label(class_index: int, grade_classes: Sequence[str]) -> str:
    """Map class index to grade label string, or return string of index if out of bounds.
    
    Converts classification network output (class index) to human-readable grade label.
    Safely handles invalid indices by falling back to string representation of index.
    
    Args:
        class_index: Integer class index from classifier output.
        grade_classes: Sequence of grade label strings (e.g., ['Good', 'Fair', 'Poor']).
        
    Returns:
        Grade label string if index is valid, otherwise string representation of class_index.
    """
    if 0 <= class_index < len(grade_classes):
        return grade_classes[class_index]
    return str(class_index)


def load_font(font_path: Optional[str], font_size: int, config: PredictConfig) -> ImageFont.ImageFont:
    """Load font for drawing text, falling back to default if unavailable.
    
    Attempts to load specified font file, with fallback chain:
    1. Load specified font_path if provided
    2. Load default font from config settings
    3. Fall back to PIL default bitmap font if all files unavailable
    
    Args:
        font_path: Optional path to custom TrueType font file.
        font_size: Font size in pixels for rendering text.
        config: PredictConfig containing default font settings.
        
    Returns:
        PIL ImageFont object ready for text rendering.
    """
    if font_path:
        return ImageFont.truetype(font_path, font_size)

    try:
        return ImageFont.truetype(config.default_font_name, font_size)
    except OSError:
        return ImageFont.load_default()


def draw_grade_label(
    image: Image.Image,
    label: str,
    font_path: Optional[str],
    font_size: int,
    config: PredictConfig,
) -> Image.Image:
    """Draw grade label text on image with white background box in upper-left corner.
    
    Creates a copy of the image and overlays a grade label with styling:
    - White rounded rectangle background for contrast
    - Black text in upper-left corner
    - Configurable font and size
    
    Args:
        image: PIL Image object to draw on (not modified in-place).
        label: Text label to draw (e.g., 'Good', 'Fair', 'Poor').
        font_path: Optional path to custom font file.
        font_size: Font size in pixels.
        config: PredictConfig with default font settings.
        
    Returns:
        New PIL Image with grade label drawn on it.
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = load_font(font_path, font_size, config)

    # Calculate padding and text bounding box
    padding = max(font_size // 4, 6)
    text_bbox = draw.textbbox((padding, padding), label, font=font)
    left, top, right, bottom = text_bbox
    # Create box around text with padding
    box = (left - padding, top - padding, right + padding, bottom + padding)

    # Draw white rounded rectangle as background for text readability
    draw.rounded_rectangle(box, radius=padding, fill=(255, 255, 255))
    # Draw black text on white background
    draw.text((padding, padding), label, font=font, fill=(0, 0, 0))
    return result


def resolve_save_path(
    source_path: Path,
    input_path: Path,
    output_arg: Optional[str],
    suffix: str,
    config: PredictConfig,
) -> Path:
    """Determine output file path based on input type and output argument.
    
    Handles three scenarios:
    1. Single file, no output arg: save in same dir as input with suffix
    2. Single file, output is explicit file path: use that path
    3. Single file, output is directory: save in that directory
    4. Batch (directory input), output is dir: save in that directory
    5. Batch, no output arg: use config default output directory
    
    Args:
        source_path: Path to source image being processed.
        input_path: Original input argument (file or directory).
        output_arg: User-provided output path argument (None for default).
        suffix: String appended to output filename (e.g., '_pred').
        config: PredictConfig with extension and default dir settings.
        
    Returns:
        Full output file path where result should be saved.
    """
    if input_path.is_file():
        if output_arg is None:
            return source_path.with_name(f"{source_path.stem}{suffix}{config.output_extension}")

        output_path = Path(output_arg).expanduser()
        if output_path.suffix:
            return output_path
        return output_path / f"{source_path.stem}{suffix}{config.output_extension}"

    # Directory input: save all results in output directory
    output_dir = Path(output_arg).expanduser() if output_arg else config.default_output_dir(input_path)
    return output_dir / f"{source_path.stem}{suffix}{config.output_extension}"


def predict_one_image(
    predictor: AmmoniaNetInferencer,
    image_path: Path,
    count: bool,
    seg_classes: Sequence[str],
) -> Tuple[Image.Image, int]:
    """Run segmentation inference on a single image.
    
    Loads image file and passes it through AmmoniaNet for prediction,
    which outputs both segmentation visualization and classification grade.
    
    Args:
        predictor: Initialized AmmoniaNetInferencer for model inference.
        image_path: Path to input image file.
        count: If True, compute and print pixel-level statistics.
        seg_classes: List of segmentation class names for statistics output.
        
    Returns:
        Tuple of (result_image, class_index) where:
        - result_image: PIL Image with segmentation visualization
        - class_index: Integer grade classification from model
    """
    with Image.open(image_path) as image:
        image = image.copy()
    return predictor.detect_image(image, count=count, name_classes=seg_classes)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for consistency and correctness.
    
    Performs validation checks:
    - Font size must be positive integer
    - Segmentation class list cannot be empty
    - Grading class list cannot be empty
    - When input is directory, output must also be directory (not file path)
    
    Args:
        args: Parsed arguments namespace from ArgumentParser.
        
    Raises:
        ValueError: If any validation check fails.
    """
    if args.font_size <= 0:
        raise ValueError("--font-size must be a positive integer.")
    if not args.seg_classes:
        raise ValueError("--seg-classes cannot be empty.")
    if not args.grade_classes:
        raise ValueError("--grade-classes cannot be empty.")

    input_path = Path(args.input_path).expanduser()
    if input_path.is_dir() and args.output and Path(args.output).expanduser().suffix:
        raise ValueError("--output must be a directory when input_path is a directory.")


def main() -> None:
    """Main entry point for batch or single-image prediction pipeline.
    
    Orchestrates complete prediction workflow:
    1. Load configuration and parse command-line arguments
    2. Validate arguments for consistency
    3. Collect image paths from input (single file or directory)
    4. Initialize AmmoniaNet inferencer with specified model
    5. For each image:
       a. Run segmentation and classification inference
       b. Optionally draw grade label on output
       c. Determine output path and save result image
    6. Print summary of saved predictions
    
    Supports:
    - Single image or batch directory processing
    - GPU or CPU inference
    - Custom output directories or same-directory saves
    - Pixel counting and statistics
    - Grade label visualization with customizable fonts
    """
    # Load configuration and parse arguments
    config = get_predict_config()
    args = parse_args(config)
    validate_args(args)

    # Collect image paths from input
    input_path = Path(args.input_path).expanduser()
    image_paths = collect_image_paths(input_path, config)

    # Initialize predictor with command-line overrides
    # Create inference config by updating specified parameters
    inference_config = replace(
        config.inference,
        model_path=args.model_path,
        mix_type=args.mix_type,
        cuda=not args.cpu,
    )
    predictor = AmmoniaNetInferencer(config=inference_config)
    # Validate that enough class names provided for statistics if enabled
    if args.count and len(args.seg_classes) < predictor.num_classes:
        raise ValueError(
            f"--seg-classes expects at least {predictor.num_classes} names when --count is enabled."
        )

    # Setup progress display and output list
    # Use tqdm progress bar for batch processing, simple list for single image
    iterator = tqdm(image_paths, desc="Predicting", unit="image") if len(image_paths) > 1 else image_paths
    saved_paths: List[Path] = []

    # Process each image
    for image_path in iterator:
        # Run inference on image to get segmentation result and grade classification
        result_image, class_index = predict_one_image(
            predictor,
            image_path=image_path,
            count=args.count,
            seg_classes=args.seg_classes,
        )

        # Draw grade label if requested
        # Converts class index to human-readable label
        grade_label = resolve_grade_label(class_index, args.grade_classes)
        if args.draw_grade_label:
            result_image = draw_grade_label(
                result_image,
                label=grade_label,
                font_path=args.font_path,
                font_size=args.font_size,
                config=config,
            )

        # Save result image
        # Determine output path based on input mode and arguments
        save_path = resolve_save_path(
            source_path=image_path,
            input_path=input_path,
            output_arg=args.output,
            suffix=args.suffix,
            config=config,
        )
        # Create parent directories if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(save_path)
        saved_paths.append(save_path)

    # Print summary
    # Show different messages for single vs. batch processing
    if len(saved_paths) == 1:
        print(f"Saved prediction to: {saved_paths[0]}")
    else:
        print(f"Saved {len(saved_paths)} predictions to: {saved_paths[0].parent}")


if __name__ == "__main__":
    main()
