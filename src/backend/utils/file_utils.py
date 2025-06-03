# backend/utils/file_utils.py

import json
import os
import shutil
import tempfile


def atomic_write(file_path, data, mode="w"):
    """
    Writes data to a file atomically.

    Args:
        file_path (str): Path to the file.
        data (dict or list or str): Data to write.
        mode (str, optional): File mode. Defaults to 'w'.
    """
    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(mode, dir=dir_name, delete=False) as tmp_file:
        if isinstance(data, (dict, list)):
            json.dump(data, tmp_file, indent=4)
        else:
            tmp_file.write(data)
        temp_name = tmp_file.name
    shutil.move(temp_name, file_path)


def read_json(file_path):
    """
    Reads JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)
