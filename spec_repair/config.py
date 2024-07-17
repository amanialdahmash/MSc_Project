import os.path

PATH_TO_CLI = os.path.expanduser("/Users/amanialdahmash/Desktop/proj/spectra-cli.jar")
PATH_TO_CORES = os.path.expanduser(
    "/Users/amanialdahmash/Desktop/SpecRepair/easy-downloads/spectra_unrealizable_cores.jar"
)
PATH_TO_ILASP = os.path.expanduser("/usr/local/bin/ILASP")
PATH_TO_FASTLAS = os.path.expanduser("~/Tools/bin/FastLAS")
PRINT_CS = False
FASTLAS = False  # TODO: modify into enum (inductive ASP tool)
RESTORE_FIRST_HYPOTHESIS = True

# This determines the paths for running clingo and ILASP and whether to use
# Windows Subsystem for Linux (WSL):
SETUP_DICT = {
    "wsl": False,
    "clingo": "clingo",
    "ILASP": PATH_TO_ILASP,
    "FastLAS": PATH_TO_FASTLAS,
    "ltlfilt": "ltlfilt",
}

PROJECT_PATH: str = os.path.expanduser("~/Desktop/SpecRepair")
GENERATE_MULTIPLE_TRACES = False

# Violation Listening Configurations
LOG_FOLDER = "/Users/tg4018/eclipse-workspace/PhD/Lift"

# TODO: add these in a config class
MAX_ASP_HYPOTHESES = 10

# For testing and statistics
STATISTICS: bool = True
MANUAL: bool = True

# Configuration of Learning
WEAKENING_TO_JUSTICE = True
