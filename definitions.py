from pathlib import Path

REPOSITORY_NAME = "mur_pysim"


# The function below will always return the project root assuming that the repository name
# is mur_pysim. It does this by only selecting the characters in the path from index 0
# up to the end of string of REPOSITORY_NAME. This means that if you are running a file
# with path of "mur_pysim/simulation/track_loader.py", you will always get "mur_pysim".
def get_project_root_wherever_you_are(path):
    return path[0:(path.find(REPOSITORY_NAME) + len(REPOSITORY_NAME))]


PROJECT_ROOT = get_project_root_wherever_you_are(Path(__file__).cwd().__str__())
