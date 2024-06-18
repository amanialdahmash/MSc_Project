import re
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log_folder = '/Users/tg4018/eclipse-workspace/PhD/Lift'


class ViolationListener(FileSystemEventHandler):
    def __init__(self, process_function):
        super().__init__()
        self.process_function = process_function

    def on_created(self, event):
        log_file_path = event.src_path
        if event.is_directory or not is_valid_log_file_name(log_file_path):
            print("It's not a new log file")
            return

        # Read the contents of the log file
        with open(log_file_path, 'r') as file:
            log_contents = file.read()

        # Call the provided function on the log contents
        result = self.process_function(log_contents)

        send_result_to_another_program(result)


def start_listening_for_violations(process_function):
    event_handler = ViolationListener(process_function)
    observer = Observer()
    observer.schedule(event_handler, log_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def send_result_to_another_program(result):
    # Send the result to another program
    # Replace with your actual implementation
    print(f"Sending result to another program: {result}")


def is_valid_log_file_name(file_name):
    """
    Checking whether the given file name is a log file produced by the Spectra synthesiser,
    given the name.
    :param file_name: name of the possible log file
    :return:
    """
    pattern = r".*_log_\d{2}\.\d{2}\.\d{4}_\d{2}_\d{2}_\d{2}\.txt$"
    return re.match(pattern, file_name) is not None
