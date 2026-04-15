from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

from config.config import DatasetLayoutConfig, PrepareDatasetConfig, get_prepare_dataset_config


@dataclass(frozen=True)
class DatasetSplit:
    """Container for dataset split information.
    
    Stores sample IDs partitioned into trainval, train, validation, and test sets
    for cross-validation or train/test evaluation.
    
    Attributes:
        trainval: List of sample IDs used for training and validation combined.
        train: List of sample IDs used for training.
        val: List of sample IDs used for validation.
        test: List of sample IDs held out for testing.
    """
    trainval: List[str]
    train: List[str]
    val: List[str]
    test: List[str]


def parse_args(config: PrepareDatasetConfig) -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation.
    
    Args:
        config: PrepareDatasetConfig instance with default values.
        
    Returns:
        Parsed arguments with dataset_root_path and split ratio parameters.
    """
    parser = argparse.ArgumentParser(
        description="Generate segmentation split files and validate label masks.",
    )
    parser.add_argument(
        "dataset_root_path",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--trainval-percent",
        type=float,
        default=config.trainval_percent,
        help="Fraction of samples used for trainval. Default: 1.0",
    )
    parser.add_argument(
        "--train-percent",
        type=float,
        default=config.train_percent,
        help="Fraction of trainval samples assigned to train. Default: 0.9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.seed,
        help="Random seed used for dataset split generation. Default: 0",
    )
    return parser.parse_args()


def resolve_dataset_dirs(dataset_root_path: str, config: DatasetLayoutConfig) -> Dict[str, Path]:
    """Resolve and validate dataset directory paths.
    
    Validates that required directories exist and creates the split directory if needed.
    
    Args:
        dataset_root_path: Root path to the dataset.
        config: Dataset layout configuration.
        
    Returns:
        Dictionary with keys: 'dataset_root', 'segmentation_dir', 'split_dir'.
        
    Raises:
        FileNotFoundError: If dataset root or segmentation directory doesn't exist.
    """
    # Get configured directory paths
    dataset_root = config.dataset_root(dataset_root_path)
    segmentation_dir = config.segmentation_dir(dataset_root_path)
    split_dir = config.split_dir(dataset_root_path)

    # Validate existence of required directories
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not segmentation_dir.exists():
        raise FileNotFoundError(f"{config.segmentation_dir_name} directory not found: {segmentation_dir}")

    # Create split directory if it doesn't exist
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_root": dataset_root,
        "segmentation_dir": segmentation_dir,
        "split_dir": split_dir,
    }


def collect_segmentation_ids(segmentation_dir: Path, config: DatasetLayoutConfig) -> List[str]:
    """Collect all segmentation mask IDs from the segmentation directory.
    
    Returns sample IDs sorted in alphabetical order.
    
    Args:
        segmentation_dir: Path to the segmentation directory.
        config: Dataset layout configuration.
        
    Returns:
        Sorted list of sample IDs (filenames without extension).
    """
    # Collect all files with the configured mask extension and extract their stem (filename without extension)
    return sorted(
        path.stem
        for path in segmentation_dir.iterdir()
        if path.suffix.lower() == config.mask_extension
    )


def build_dataset_split(
    sample_ids: List[str],
    trainval_percent: float,
    train_percent: float,
    seed: int,
) -> DatasetSplit:
    """Build dataset split (train/val/test) from sample IDs.
    
    Partitions samples into trainval set (further split into train and val) and test set.
    
    Args:
        sample_ids: List of all sample IDs.
        trainval_percent: Fraction of samples to use for trainval (rest go to test).
        train_percent: Fraction of trainval samples to use for train (rest go to val).
        seed: Random seed for reproducibility.
        
    Returns:
        DatasetSplit with train, val, trainval, and test sample lists.
        
    Raises:
        ValueError: If split ratios are not between 0 and 1.
    """
    # Validate split ratio parameters
    validate_split_ratio(trainval_percent, "trainval_percent")
    validate_split_ratio(train_percent, "train_percent")

    # Initialize random generator with seed for reproducible splitting
    rng = random.Random(seed)
    total = len(sample_ids)
    # Calculate number of samples for each split
    trainval_count = int(total * trainval_percent)
    train_count = int(trainval_count * train_percent)

    # Randomly select samples for trainval set (rest go to test)
    trainval_ids = set(rng.sample(sample_ids, trainval_count))
    # Further split trainval into train and val
    train_ids = set(rng.sample(sorted(trainval_ids), train_count))

    # Partition all samples into appropriate splits
    trainval = []
    train = []
    val = []
    test = []
    for sample_id in sample_ids:
        if sample_id in trainval_ids:
            trainval.append(sample_id)
            if sample_id in train_ids:
                train.append(sample_id)
            else:
                val.append(sample_id)
        else:
            test.append(sample_id)

    return DatasetSplit(trainval=trainval, train=train, val=val, test=test)


def validate_split_ratio(value: float, field_name: str) -> None:
    """Validate that a split ratio is between 0 and 1.
    
    Args:
        value: Ratio value to validate.
        field_name: Name of the field for error messages.
        
    Raises:
        ValueError: If value is not between 0 and 1.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1, got {value}.")


def write_split_files(split_dir: Path, split: DatasetSplit, config: DatasetLayoutConfig) -> None:
    """Write dataset split information to text files.
    
    Creates separate text files for trainval, train, val, and test splits,
    each containing one sample ID per line.
    
    Args:
        split_dir: Directory to write split files into.
        split: DatasetSplit containing sample lists.
        config: Dataset configuration with filename settings.
    """
    # Write each split to its configured output file
    write_split_file(split_dir / config.trainval_split_filename, split.trainval)
    write_split_file(split_dir / config.train_split_filename, split.train)
    write_split_file(split_dir / config.val_split_filename, split.val)
    write_split_file(split_dir / config.test_split_filename, split.test)


def write_split_file(file_path: Path, sample_ids: List[str]) -> None:
    """Write sample IDs to a text file, one per line.
    
    Args:
        file_path: Path to the output file.
        sample_ids: List of sample IDs to write.
    """
    # Write all sample IDs with one ID per line
    file_path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")


def validate_masks(
    segmentation_dir: Path,
    sample_ids: List[str],
    config: DatasetLayoutConfig,
) -> np.ndarray:
    """Validate all segmentation masks and collect pixel value statistics.
    
    Checks that all masks exist, are grayscale or 8-bit, and collects histograms
    of pixel values to detect potential data format issues.
    
    Args:
        segmentation_dir: Directory containing mask files.
        sample_ids: List of sample IDs to validate.
        config: Dataset configuration.
        
    Returns:
        Histogram array showing count of each pixel value (0-255).
        
    Raises:
        ValueError: If a mask file doesn't exist or has incorrect format.
    """
    # Initialize histogram array for all possible pixel values
    class_counts = np.zeros(256, dtype=np.int64)

    # Process each mask file
    for sample_id in tqdm(sample_ids):
        # Construct mask file path
        mask_path = segmentation_dir / f"{sample_id}{config.mask_extension}"
        if not mask_path.exists():
            raise ValueError(
                "Label image was not detected. Please check the file path and make sure the "
                f"extension is {config.mask_extension}: {mask_path}"
            )

        # Load mask image and convert to numpy array
        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask, dtype=np.uint8)

        # Warn if mask is not grayscale or 8-bit color
        if mask_array.ndim > 2:
            print(
                f"The shape of label image {mask_path.name} is {mask_array.shape}. "
                "It is not a grayscale image or 8-bit color image."
            )
            print(
                "Label images must be grayscale or 8-bit color images. "
                "Each pixel value should represent the class index."
            )

        # Accumulate pixel value histogram
        class_counts += np.bincount(mask_array.reshape(-1), minlength=256)

    return class_counts


def print_class_statistics(class_counts: np.ndarray) -> None:
    """Print pixel value counts in a formatted table.
    
    Args:
        class_counts: Histogram array of pixel value counts.
    """
    print("Printing pixel values and their corresponding counts.")
    print("-" * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print("-" * 37)
    # Print all pixel values that appear at least once
    for class_index, count in enumerate(class_counts):
        if count > 0:
            print("| %15s | %15s |" % (str(class_index), str(count)))
            print("-" * 37)


def print_dataset_warnings(class_counts: np.ndarray, config: DatasetLayoutConfig) -> None:
    """Print warnings if potential dataset format issues are detected.
    
    Detects common mistakes such as using 0/255 binary labels instead of 0/1,
    or having all background pixels with no foreground objects.
    
    Args:
        class_counts: Histogram array of pixel value counts.
        config: Dataset configuration for directory names.
    """
    # Check for 0/255 binary labeling (incorrect format)
    if class_counts[255] > 0 and class_counts[0] > 0 and np.sum(class_counts[1:255]) == 0:
        print("Detected that pixel values in the labels only contain 0 and 255. The data format is incorrect.")
        print("For binary segmentation, background should be 0 and target should be 1.")
    # Check for all background pixels (no foreground objects)
    elif class_counts[0] > 0 and np.sum(class_counts[1:]) == 0:
        print("Detected that the labels only contain background pixels. Please check the dataset format.")

    # Print file format expectations
    print(
        f"Images in {config.image_dir_name} should be {config.image_extension} files, "
        f"and masks in {config.segmentation_dir_name} should be {config.mask_extension} files."
    )


def main() -> None:
    """Main entry point for dataset preparation.
    
    Orchestrates the complete dataset preparation pipeline:
    1. Parse command-line arguments and load configuration
    2. Validate dataset directory structure
    3. Collect sample IDs from segmentation masks
    4. Generate train/val/test splits and save to text files
    5. Validate all masks and collect statistics
    6. Print warnings about potential data format issues
    """
    # Load configuration and parse command-line arguments
    config = get_prepare_dataset_config()
    args = parse_args(config)
    random.seed(args.seed)

    # Resolve and validate dataset directory paths
    dataset_config = config.dataset
    dataset_dirs = resolve_dataset_dirs(args.dataset_root_path, dataset_config)
    segmentation_dir = dataset_dirs["segmentation_dir"]
    split_dir = dataset_dirs["split_dir"]
    # Collect all sample IDs from segmentation masks
    sample_ids = collect_segmentation_ids(segmentation_dir, dataset_config)

    # Generate and write dataset split files
    print(f"Generate txt in {dataset_config.split_root_dir_name}.")
    split = build_dataset_split(
        sample_ids=sample_ids,
        trainval_percent=args.trainval_percent,
        train_percent=args.train_percent,
        seed=args.seed,
    )
    write_split_files(split_dir, split, dataset_config)

    # Print split statistics
    print("train and val size", len(split.trainval))
    print("train size", len(split.train))
    print(f"Generate txt in {dataset_config.split_root_dir_name} done.")

    # Validate all masks and collect statistics
    print("Check datasets format, this may take a while.")
    class_counts = validate_masks(segmentation_dir, sample_ids, dataset_config)
    # Print statistics and potential warnings
    print_class_statistics(class_counts)
    print_dataset_warnings(class_counts, dataset_config)


if __name__ == "__main__":
    main()
