from fire import Fire

from whisperplus import __version__ as whisperplus_version

whisperplus_app = {"version": whisperplus_version}


def app():
    Fire(whisperplus_app)


if __name__ == "__main__":
    app()
