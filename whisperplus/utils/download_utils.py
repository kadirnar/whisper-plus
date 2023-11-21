from pathlib import Path

from pytube import YouTube


def download_and_convert_to_mp3(url: str, output_path: str = "output", filename: str = "test") -> str:
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=output_path, filename=filename)

    new_file = Path(out_file).with_suffix('.mp3')

    if new_file.exists():
        new_file.unlink()

    Path(out_file).rename(new_file)
    return str(new_file)
