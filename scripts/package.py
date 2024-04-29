import subprocess


def run_command(command):
    """
    Executes a given command using the subprocess module.
    
    Args:
        command (str): The command to be executed.
    """
    subprocess.run(command, shell=True, check=True)


def main():
    """
    Main function to package and upload a Python package using setup.py and twine.
    """
    sdist_command = "python setup.py sdist"
    twine_command = "twine upload dist/*"

    run_command(sdist_command)
    run_command(twine_command)


if __name__ == "__main__":
    main()
