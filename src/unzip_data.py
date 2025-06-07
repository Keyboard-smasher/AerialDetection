import sys
import zipfile
from pathlib import Path


def unzip_file(zip_path, extract_to):
    """Unzip a file using Python's built-in zipfile module"""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    print(f"Unzipping {zip_path} to {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted {len(zip_ref.namelist())} files")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unzip_data.py <zip_file> <output_dir>")
        sys.exit(1)

    zip_file = sys.argv[1]
    output_dir = sys.argv[2]
    unzip_file(zip_file, output_dir)
