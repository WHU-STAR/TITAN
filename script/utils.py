import time
import os
from data_loader import get_data_paths


class Timer:
    """
    Timer class to calculate time
    """

    def __init__(self):
        self.start = None

    def tick(self):
        self.start = time.perf_counter()

    def tuck(self, message: str = ""):
        if self.start is None:
            raise RuntimeError("tick() must be called before tuck()")
        # calculate time in seconds
        elapsed = time.perf_counter() - self.start
        if message != "":
            print(f"{message} time: {elapsed:.6f}s")
        return elapsed


def generate_path_script(destdir: str, dataset_name: str):
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    data_paths = get_data_paths(dataset_name)
    # create a new file under destdir called get_data_path.sh
    query_path = data_paths["query"]
    groundtruth_path = data_paths["groundtruth"]
    file_name = os.path.join(destdir, "get_data_path.sh")
    with open(file_name, "w") as f:
        f.write(f'echo "{query_path}"\n')
        f.write(f'echo "{groundtruth_path}"\n')

    os.system(f"chmod +x {file_name}")

    
def generate_help_message(param_config, partition_config):
    help_message = "Usage: python data_processor.py <dataset_name> <partition_method>\n"
    help_message += "Valid dataset names are:\n"
    for key in param_config:
        help_message += f"- {key}\n"
    help_message += "Valid partition methods are:\n"
    for key in partition_config:
        help_message += f"- {key}\n"
    return help_message


def print_help(param_config, partition_config):
    print(generate_help_message(param_config, partition_config))