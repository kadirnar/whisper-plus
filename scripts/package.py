import subprocess


def run_command(command):
    subprocess.run(command, shell=True, check=True)


def main():
    sdist_command = "python setup.py sdist"
    twine_command = "twine upload dist/*"

    run_command(sdist_command)
    run_command(twine_command)


if __name__ == "__main__":
    main()
