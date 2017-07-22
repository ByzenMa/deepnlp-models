# encoding=utf-8
import os


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r') as f:
        return f.read()
