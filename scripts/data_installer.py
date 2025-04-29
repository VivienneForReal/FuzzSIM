# -*- coding: utf-8 -*-
import kagglehub
import shutil
import os
import sys
import argparse

def install_kaggle_datasets(dataset_name, output_dir="data"):
    if not dataset_name:
        raise ValueError("Dataset name cannot be None or empty")

    if dataset_name == 'iris':
        path = kagglehub.dataset_download("uciml/iris")
    elif dataset_name == 'penguins':
        path = kagglehub.dataset_download("muhammadnoumangul/penguins-dataset")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print("Downloaded to:", path)

    # Ensure output directory exists
    output_path = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Move contents to the specified output directory
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(output_path, item)
        shutil.move(src, dst)

    print(f"Dataset '{dataset_name}' moved to '{output_dir}' successfully.")

def main():
    parser = argparse.ArgumentParser(description="Install a dataset from KaggleHub.")
    parser.add_argument("--dataset-name", required=True, help="The name of the dataset to install.")
    parser.add_argument("--output-dir", default="data", help="The directory to save the dataset.")

    args = parser.parse_args()
    try:
        install_kaggle_datasets(args.dataset_name, args.output_dir)
    except ValueError as ve:
        print(f"\nValueError: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
