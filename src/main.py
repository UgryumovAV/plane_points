import argparse
import os.path
from glob import glob
from pathlib import Path

from src.tools.extract_tools import extract_walls_from_image


def main(
        source_dir_path: str,
        save_dir_path: str,
        ocr_flag: bool = True
) -> None:
    root_path = Path(__file__).resolve(strict=True).parent.parent

    if not os.path.isabs(source_dir_path):
        source_dir_path = os.path.join(root_path, Path(source_dir_path))

    if not os.path.isabs(save_dir_path):
        save_dir_path = os.path.join(root_path, Path(save_dir_path))

    if not os.path.exists(save_dir_path):
        Path(save_dir_path).mkdir(parents=True, exist_ok=True)

    images_paths = glob(source_dir_path + "/*")
    print(f"Source directory: {source_dir_path}. Number of planes: {len(images_paths)}")

    for image_path in images_paths:
        extract_walls_from_image(
            image_path=image_path,
            save_dir_path=save_dir_path,
            ocr_flag=ocr_flag,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract walls from a floor plan images.')
    parser.add_argument(
        '-d',
        '--source_directory',
        default="data/planes",
        help='Path to the dir of input floor plan images.'
    )
    parser.add_argument(
        '-s',
        '--save_directory',
        default="data/results",
        help='Path to the dir to save results.'
    )
    parser.add_argument(
        '-o',
        '--ocr',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='OCR flag usage.'
    )
    args = parser.parse_args()

    main(
        source_dir_path=args.source_directory,
        ocr_flag=args.ocr,
        save_dir_path=args.save_directory
    )

    # python -m src.main
