import os
import shutil
import random

base_path = "/mnt/sda1/du_an1/data/val/waves"

new_folder = "/mnt/sda1/du_an1/data/val/output/enroll"

os.makedirs(new_folder, exist_ok=True)

if os.path.exists(base_path):
    all_items = os.listdir(base_path)
    enrollments = [
        d
        for d in all_items
        if os.path.isdir(os.path.join(base_path, d))
        and "aligned_wav2vec2" not in d
        and "output" not in d
        and d != "noise"
    ]

    for enrollment in enrollments:
        enrollment_path = os.path.join(base_path, enrollment)

        files = [f for f in os.listdir(enrollment_path) if f.endswith(".wav")]

        if not files:
            continue

        if len(files) >= 10:
            selected_files = random.sample(files, 10)
        else:
            selected_files = files

        enroll_subfolder = os.path.join(new_folder, enrollment)
        os.makedirs(enroll_subfolder, exist_ok=True)

        for file in selected_files:
            src = os.path.join(enrollment_path, file)
            dst = os.path.join(enroll_subfolder, file)
            shutil.copy(src, dst)
            print(f"Copied {file} to {enroll_subfolder}")

print("Random selection and copying completed.")