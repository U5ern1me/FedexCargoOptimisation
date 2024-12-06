import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizer import visualize
from utils.io import load_config

config = load_config(os.path.join(os.path.dirname(__file__), "visualize.config"))

if __name__ == "__main__":
    output_path = config["output folder"]
    input_folder = config["input folder"]
    visualize(output_path, input_folder)
