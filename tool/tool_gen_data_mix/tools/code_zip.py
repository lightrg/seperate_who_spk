import zipfile
from pathlib import Path
from typing import List

def zip_directories(source_dirs: List[Path], output_zip: Path) -> None:

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for source_dir in source_dirs:
            if not source_dir.exists():
                print(f"Cảnh báo: Thư mục {source_dir} không tồn tại. Bỏ qua.")
                continue

            for file_path in source_dir.rglob('*'):

                arcname = Path(source_dir.name) / file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)

    print(f"Đã tạo file zip tại: {output_zip}")

if __name__ == "__main__":

    BASE_PATH = Path("/run/media/phuong/HDD/")
    SOURCES = [

        BASE_PATH / "data"
    ]
    OUTPUT = BASE_PATH / "data.zip"

    zip_directories(SOURCES, OUTPUT)