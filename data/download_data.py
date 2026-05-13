"""
Download Active Matter dataset from The Well benchmark.
"""

import argparse
from the_well.utils.download import well_download


def download_active_matter(base_path: str):
    """Download train/valid/test splits for Active Matter dataset."""

    print(f"Downloading Active Matter dataset to {base_path}")

    for split in ["train", "valid", "test"]:
        print(f"  Downloading {split} split...")
        well_download(base_path=base_path, dataset="active_matter", split=split)

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Active Matter dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to store the dataset"
    )
    args = parser.parse_args()

    download_active_matter(args.base_path)
