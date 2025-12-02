import os
import re

def load_text_from_folder(folder_path):
    all_lines = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Remove XML-like tags
                        line = re.sub(r"</?[^>]+>", "", line).strip()


                        if isinstance(line, str):
                            all_lines.append(line)

    return all_lines     

