import os

def load_text_from_folder(folder_path):
    lines = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                lines.extend([line.strip() for line in f if line.strip()])
    return lines

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
