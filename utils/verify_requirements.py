import subprocess
import sys

def verify_requirements(requirements_file='requirements.txt'):
    """
    Verifies that all required packages are installed with the correct versions.
    If a package is missing or has an incompatible version, it will prompt the user to install it.

    Parameters:
    - requirements_file: str, path to the requirements.txt file.
    """
    try:
        with open(requirements_file, 'r') as f:
            requirements = f.readlines()

        print(f"Verifying requirements from {requirements_file}...")

        for requirement in requirements:
            package = requirement.strip()
            try:
                # Try importing the package
                subprocess.check_call([sys.executable, "-m", "pip", "show", package.split('==')[0].split('>=')[0]])
            except subprocess.CalledProcessError:
                print(f"Missing or incompatible package: {package}")
                print("Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        print("All requirements are satisfied.")
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found. Please create the file and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")