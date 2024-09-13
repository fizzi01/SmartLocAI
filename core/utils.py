import zipfile
from io import BytesIO
import os


def create_zip(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_path in files:
            with open(file_path, "rb") as f:
                zip_file.writestr(os.path.basename(file_path), f.read())
    zip_buffer.seek(0)
    return zip_buffer


def get_all_files(directory):
    file_info = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            file_info.append({
                "File": file,
                "Percorso": filepath,
                "Directory": os.path.basename(root)
            })
    return file_info
