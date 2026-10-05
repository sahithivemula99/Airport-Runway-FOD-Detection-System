"""
Creates a downloadable ZIP archive of the complete project folder.
"""

import os
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Dell\.gemini\antigravity\scratch\fod_detection_system")
ARTIFACT_DIR = Path(r"C:\Users\Dell\.gemini\antigravity\brain\22143976-75c3-4c6c-883a-8caa0b51a8c7")

zip_dest1 = BASE_DIR / "fod_detection_system.zip"
zip_dest2 = ARTIFACT_DIR / "fod_detection_system.zip"

print(f"Archiving project from: {BASE_DIR}")

with zipfile.ZipFile(zip_dest1, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BASE_DIR):
        # Exclude pycache, git, and venv
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "venv", "env", ".idea"]]
        for file in files:
            if file.endswith(".zip") or file.endswith(".pt"):
                continue
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, BASE_DIR)
            zipf.write(file_path, arcname)

shutil.copy(zip_dest1, zip_dest2)
print(f"ZIP package created successfully at:\n- {zip_dest1}\n- {zip_dest2}")
