import os
import random
import re
import string

from spec_repair.old.case_study_translator import delete_files
from spec_repair.old.util_titus import generate_trace_asp

# Custom type definitions
Log = str
ASPTrace = str
FilePath = str


# TODO: remove this or write_file() from specification_helper.py
def write_to_file(filename: FilePath, content: str):
    with open(filename, 'w') as file:
        file.write(content)


def read_file(file_path: FilePath) -> str:
    with open(file_path, 'r') as file:
        file_content: str = file.read()
    return file_content


def is_file_format(file_path: str, file_extension: str) -> bool:
    """
    Checks if the path to the (possibly not-existent file) exists,
    then makes sure the extension is the expected one.
    :param file_path: Complete path to a file
    :param file_extension: Expected extension of the file
    :return: True if the file format is expected
    """
    directory, _ = os.path.split(file_path)
    if not os.path.exists(directory):
        return False

    _, extension = os.path.splitext(file_path)
    return extension == file_extension


def generate_filename(spectra_file, replacement, output=False):
    if output:
        spectra_file = spectra_file.replace("input", "output")
    return spectra_file.replace(".spectra", replacement)


def generate_asp_trace_to_file(start_file: str, end_file: str) -> str:
    trace_file = generate_temp_filename(".txt")
    generate_trace_asp(start_file, end_file, trace_file)
    return trace_file


def generate_temp_filename(ext):
    assert is_file_extension(ext)
    random_name = generate_random_string(length=10)
    temp_path = os.path.join('/tmp', f"{random_name}{ext}")
    return temp_path


def generate_random_string(length: int = 10) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


def is_file_extension(filename: str) -> bool:
    return bool(re.match(r"\.[a-zA-Z0-9_]+$", filename))
