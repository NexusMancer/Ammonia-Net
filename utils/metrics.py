from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.config import (
    DatasetLayoutConfig,
    MetricsOutputConfig,
    get_dataset_config,
    get_metrics_config,
)
from utils.segmentation_utils import compute_tp_fp_fn, compute_f_score, compute_f1_score


# Default class names for segmentation tasks
DEFAULT_SEGMENT_CLASS_NAMES = ("_background_", "Tongue", "toothmarks")


def f_score(
    inputs: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1,
    smooth: float = 1e-5,
    threhold: float = 0.5,
) -> torch.Tensor:
    """Compute the mean Dice/F-score over segmentation classes.
    
    Args:
        inputs: Model predictions of shape (batch_size, num_classes, height, width).
        target: Ground truth labels of shape (batch_size, height, width, channels).
        beta: Beta parameter for F-score calculation. Default is 1 for Dice score.
        smooth: Smoothing constant to avoid division by zero.
        threhold: Probability threshold for binarizing predictions.
        
    Returns:
        Mean F-score across all classes.
        
    Raises:
        ValueError: If inputs or target are not 4D tensors or batch sizes don't match.
    """
    # Validate input tensor dimensions
    if inputs.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected inputs and target to be 4D tensors, got {inputs.ndim}D and {target.ndim}D."
        )

    # Extract dimensions from input tensors
    batch_size, num_classes, height, width = inputs.size()
    target_batch_size, target_height, target_width, target_channels = target.size()
    if batch_size != target_batch_size:
        raise ValueError(
            f"Batch size mismatch between inputs ({batch_size}) and target ({target_batch_size})."
        )

    # Resize inputs to match target dimensions if necessary
    if (height, width) != (target_height, target_width):
        inputs = F.interpolate(
            inputs,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )

    # Apply softmax to get class probabilities and flatten spatial dimensions
    probabilities = torch.softmax(
        inputs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_classes),
        dim=-1,
    )
    # Flatten target tensor for easier computation
    target_flat = target.view(batch_size, -1, target_channels)

    # Binarize predictions using threshold
    predicted_mask = torch.gt(probabilities, threhold).float()
    # Calculate true positives, false positives, and false negatives
    tp, fp, fn = compute_tp_fp_fn(predicted_mask, target_flat[..., :-1])

    # Compute F-score using the beta parameter
    score = compute_f_score(tp, fp, fn, beta=beta, smooth=smooth)
    return torch.mean(score)


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Create confusion matrix from flattened ground-truth and prediction arrays.
    
    Args:
        a: Flattened ground-truth label array.
        b: Flattened prediction array.
        n: Number of classes.
        
    Returns:
        Confusion matrix of shape (n, n).
    """
    # Filter out invalid predictions (outside class range)  
    valid = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    # Encode predictions and labels into single index for fast histogram computation
    encoded = n * a[valid].astype(int) + b[valid].astype(int)
    # Count occurrences and reshape to confusion matrix
    return np.bincount(encoded, minlength=n ** 2).reshape(n, n)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    """Calculate Intersection over Union (IoU) for each class.
    
    Args:
        hist: Confusion matrix.
        
    Returns:
        Per-class IoU scores.
    """
    # IoU = TP / (TP + FP + FN)
    denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    return _safe_divide(np.diag(hist), denominator)


def per_class_PA_Recall(hist: np.ndarray) -> np.ndarray:
    """Calculate Pixel Accuracy and Recall for each class.
    
    Args:
        hist: Confusion matrix.
        
    Returns:
        Per-class Pixel Accuracy/Recall scores. Also known as Sensitivity or True Positive Rate.
    """
    # Recall = TP / (TP + FN)
    return _safe_divide(np.diag(hist), hist.sum(axis=1))


def per_class_Precision(hist: np.ndarray) -> np.ndarray:
    """Calculate Precision for each class.
    
    Args:
        hist: Confusion matrix.
        
    Returns:
        Per-class Precision scores.
    """
    # Precision = TP / (TP + FP)
    return _safe_divide(np.diag(hist), hist.sum(axis=0))


def per_Accuracy(hist: np.ndarray) -> float:
    """Calculate overall pixel accuracy.
    
    Args:
        hist: Confusion matrix.
        
    Returns:
        Overall accuracy score as a float between 0 and 1.
    """
    total = float(np.sum(hist))
    if total == 0:
        return 0.0
    # Accuracy = (TP + TN) / Total
    return float(np.sum(np.diag(hist)) / total)


def compute_mIoU(
    gt_dir,
    pred_dir,
    png_name_list,
    num_classes,
    img_dic=None,
    name_classes=None,
    dataset_config: Optional[DatasetLayoutConfig] = None,
):
    """Compute mean Intersection over Union and other segmentation metrics.
    
    Evaluates model predictions against ground truth masks and computes per-class
    and mean metrics including IoU, Pixel Accuracy, Recall, Precision, and F1-score.
    
    Args:
        gt_dir: Directory containing ground truth mask images.
        pred_dir: Directory containing prediction mask images.
        png_name_list: List of image IDs to evaluate.
        num_classes: Number of segmentation classes.
        img_dic: Optional dictionary for sample-level classification labels (gating).
        name_classes: Optional list of class names for reporting.
        dataset_config: Optional dataset configuration for file extension settings.
        
    Returns:
        Tuple of (confusion_matrix, per_class_ious, per_class_recalls, per_class_precisions, overall_accuracy).
    """
    print("Num classes", num_classes)

    # Initialize paths and configuration
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    config = dataset_config or get_dataset_config()
    class_names = _resolve_class_names(name_classes, int(num_classes))
    classification_by_image = img_dic or {}

    # Initialize confusion matrix for accumulating statistics
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    # Build paths for all ground truth and prediction images
    gt_images = [gt_root / f"{image_id}{config.mask_extension}" for image_id in png_name_list]
    pred_images = [pred_root / f"{image_id}{config.mask_extension}" for image_id in png_name_list]

    # Process each image and accumulate statistics
    for index, (image_id, gt_path, pred_path) in enumerate(zip(png_name_list, gt_images, pred_images)):
        # Load prediction mask and apply prediction gate if needed
        with Image.open(pred_path) as prediction_image:
            pred = np.asarray(prediction_image)
        pred = _apply_prediction_gate(pred, image_id, classification_by_image)

        # Load ground truth mask
        with Image.open(gt_path) as label_image:
            label = np.asarray(label_image)

        # Skip images with shape mismatch
        if label.shape != pred.shape:
            print(
                "Skipping: shape(gt) = {}, shape(pred) = {}, {}, {}".format(
                    label.shape,
                    pred.shape,
                    gt_path,
                    pred_path,
                )
            )
            continue

        # Accumulate confusion matrix from current image
        hist += fast_hist(label.reshape(-1), pred.reshape(-1), num_classes)

        # Print progress every 10 images
        if class_names is not None and index > 0 and index % 10 == 0:
            print(
                "{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%".format(
                    index,
                    len(gt_images),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist),
                )
            )

    # Compute per-class and overall metrics from confusion matrix
    ious = per_class_iu(hist)
    pa_recall = per_class_PA_Recall(hist)
    precision = per_class_Precision(hist)
    accuracy = per_Accuracy(hist)

    # Print per-class results if class names are provided
    if class_names is not None:
        for class_index, class_name in enumerate(class_names):
            f1_score = compute_f1_score(pa_recall[class_index], precision[class_index])
            print(
                "===>{}:\tIou-{}; Recall (equal to the PA)-{}; Precision-{}; F1_Score-{}".format(
                    class_name,
                    round(ious[class_index] * 100, 2),
                    round(pa_recall[class_index] * 100, 2),
                    round(precision[class_index] * 100, 2),
                    round(f1_score * 100, 2),
                )
            )

    # Print overall mean metrics
    print(
        "===> mIoU: {}; mPA: {}; Accuracy: {}".format(
            round(np.nanmean(ious) * 100, 2),
            round(np.nanmean(pa_recall) * 100, 2),
            round(accuracy * 100, 2),
        )
    )
    return hist.astype(int), ious, pa_recall, precision, accuracy


def adjust_axes(r, t, fig, axes) -> None:
    """Adjust plot axes width to accommodate text labels without clipping.
    
    Args:
        r: Renderer object for computing text dimensions.
        t: Text object whose width determines axis adjustment.
        fig: Figure object.
        axes: Axes object to adjust.
    """
    # Calculate text width in inches
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # Compute required figure width adjustment
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    # Expand x-axis limits proportionally
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(
    values,
    name_classes,
    plot_title,
    x_label,
    output_path,
    tick_font_size=12,
    plt_show=True,
):
    """Create and save a horizontal bar plot with metric values.
    
    Args:
        values: List of metric values to plot.
        name_classes: List of class names for y-axis labels.
        plot_title: Title for the plot.
        x_label: Label for x-axis.
        output_path: Path where to save the plot image.
        tick_font_size: Font size for axis ticks. Default 12.
        plt_show: Whether to display the plot. Default True.
    """
    # Setup output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create horizontal bar plot
    figure, axes = plt.subplots()
    axes.barh(range(len(values)), values, color="royalblue")
    axes.set_title(plot_title, fontsize=tick_font_size + 2)
    axes.set_xlabel(x_label, fontsize=tick_font_size)
    axes.set_yticks(range(len(values)))
    axes.set_yticklabels(name_classes, fontsize=tick_font_size)

    # Render figure and add value labels above bars
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for index, value in enumerate(values):
        # Format value display (integer or 2 decimal places)
        display_value = f" {value}" if value >= 1.0 else f" {value:.2f}"
        text = axes.text(
            value,
            index,
            display_value,
            color="royalblue",
            va="center",
            fontweight="bold",
        )
        # Adjust axes for last label to prevent clipping
        if index == len(values) - 1:
            adjust_axes(renderer, text, figure, axes)

    # Save figure
    figure.tight_layout()
    figure.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close(figure)


def show_results(
    miou_out_path,
    hist,
    IoUs,
    PA_Recall,
    Precision,
    name_classes,
    tick_font_size=12,
    metrics_config: Optional[MetricsOutputConfig] = None,
):
    """Generate visualization plots and confusion matrix for segmentation metrics.
    
    Creates bar plots for IoU, Pixel Accuracy, Recall, and Precision metrics,
    and saves a confusion matrix as CSV file.
    
    Args:
        miou_out_path: Output directory path for all generated files.
        hist: Confusion matrix data.
        IoUs: Per-class Intersection over Union scores.
        PA_Recall: Per-class Pixel Accuracy/Recall scores.
        Precision: Per-class Precision scores.
        name_classes: List of class names.
        tick_font_size: Font size for plot ticks. Default 12.
        metrics_config: Optional configuration for output filenames.
    """
    # Setup output directory and resolve class names
    output_dir = Path(miou_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = metrics_config or get_metrics_config()
    resolved_name_classes = _resolve_class_names(name_classes, len(hist))

    # Define output file paths for all metrics
    miou_plot_path = output_dir / config.miou_plot_filename
    mpa_plot_path = output_dir / config.mpa_plot_filename
    recall_plot_path = output_dir / config.recall_plot_filename
    precision_plot_path = output_dir / config.precision_plot_filename
    confusion_matrix_path = output_dir / config.confusion_matrix_filename

    # Generate and save mIoU plot
    draw_plot_func(
        IoUs,
        resolved_name_classes,
        "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100),
        "Intersection over Union",
        miou_plot_path,
        tick_font_size=tick_font_size,
        plt_show=True,
    )
    print(f"Save mIoU out to {miou_plot_path}")

    # Generate and save Pixel Accuracy plot
    draw_plot_func(
        PA_Recall,
        resolved_name_classes,
        "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100),
        "Pixel Accuracy",
        mpa_plot_path,
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print(f"Save mPA out to {mpa_plot_path}")

    # Generate and save Recall plot
    draw_plot_func(
        PA_Recall,
        resolved_name_classes,
        "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100),
        "Recall",
        recall_plot_path,
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print(f"Save Recall out to {recall_plot_path}")

    # Generate and save Precision plot
    draw_plot_func(
        Precision,
        resolved_name_classes,
        "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
        "Precision",
        precision_plot_path,
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print(f"Save Precision out to {precision_plot_path}")

    # Export confusion matrix as CSV with class names as headers and row labels
    with confusion_matrix_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        rows = [[" "] + [str(class_name) for class_name in resolved_name_classes]]
        for index, class_name in enumerate(resolved_name_classes):
            rows.append([class_name] + [str(value) for value in hist[index]])
        writer.writerows(rows)
    print(f"Save confusion_matrix out to {confusion_matrix_path}")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Perform safe division returning zeros where denominator is zero.
    
    Args:
        numerator: Numerator array.
        denominator: Denominator array.
        
    Returns:
        Division result with zeros where division by zero would occur.
    """
    # Convert inputs to float64 for precision
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    # Perform division, outputting zeros where denominator is zero
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _resolve_class_names(
    name_classes: Optional[Sequence[str]],
    num_classes: int,
) -> Tuple[str, ...]:
    """Resolve class names, using defaults if not provided.
    
    Args:
        name_classes: Optional sequence of class names.
        num_classes: Expected number of classes.
        
    Returns:
        Tuple of class names.
        
    Raises:
        ValueError: If provided class names count doesn't match num_classes.
    """
    # Use default class names if not provided
    if name_classes is None:
        if num_classes == len(DEFAULT_SEGMENT_CLASS_NAMES):
            return DEFAULT_SEGMENT_CLASS_NAMES
        # Generate generic class names for other counts
        return tuple(f"class_{index}" for index in range(num_classes))

    # Validate and return provided class names
    resolved_names = tuple(str(class_name) for class_name in name_classes)
    if len(resolved_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, but got {len(resolved_names)}."
        )
    return resolved_names


def _apply_prediction_gate(
    pred: np.ndarray,
    image_id: str,
    classification_by_image: Mapping[str, int],
) -> np.ndarray:
    """Apply prediction gating to mask predictions based on sample classification.
    
    For specific sample types (e.g., 'None' class), modifies predictions by converting
    class 2 predictions to class 1 while preserving other predictions.
    
    Args:
        pred: Prediction mask array.
        image_id: Image identifier used to look up classification.
        classification_by_image: Mapping from image IDs to classification labels.
        
    Returns:
        Modified prediction array with gating applied if needed.
    """
    # Extract sample key from image path
    sample_key = Path(str(image_id)).stem
    gating_label = classification_by_image.get(sample_key)
    # Check if this is a 'None' class sample (encoded in filename)
    is_none_sample = sample_key.split("-")[0] == "None"

    # Apply prediction gate: convert class 2 to class 1 for negative or None samples
    # This maintains backward compatibility with filename-based encoding
    if gating_label == 0 or is_none_sample:
        return np.where(pred != 2, pred, 1)
    return pred
