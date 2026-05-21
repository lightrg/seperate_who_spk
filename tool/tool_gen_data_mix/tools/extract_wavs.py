import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_wav_files(source_dir: str, dest_dir: str):

    src_path = Path(source_dir)
    dst_path = Path(dest_dir)

    if not src_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    dst_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting wav files to: {dst_path}")

    wav_files = list(src_path.rglob("*.wav"))
    total_files = len(wav_files)

    if total_files == 0:
        logger.warning("No wav files found.")
        return

    logger.info(f"Found {total_files} wav files. Starting copy...")

    success_count = 0
    for idx, wav_file in enumerate(wav_files):
        try:

            relative_path = wav_file.relative_to(src_path)
            dest_file = dst_path / relative_path

            dest_file.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(wav_file, dest_file)
            success_count += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
                logger.info(f"Progress: {idx + 1}/{total_files} files copied")

        except Exception as e:
            logger.error(f"Failed to copy {wav_file.name}: {e}")

    logger.info("=" * 40)
    logger.info(f"COMPLETED: Copied {success_count}/{total_files} files")
    logger.info(f"Destination: {dst_path}")
    logger.info("=" * 40)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract WAV files from source directory preserving structure")
    parser.add_argument("source", help="Source directory containing meetings")
    parser.add_argument("destination", help="Destination directory to copy WAV files to")

    args = parser.parse_args()

    extract_wav_files(args.source, args.destination)