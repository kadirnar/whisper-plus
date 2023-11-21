import os


def download_video(url: str, output_path: str = "output", filename: str = "test") -> str:
    """
    Downloads a video from the given URL and converts it to an MP3 file.

    Args:
        url (str): The URL of the video to download.

    Returns:
        str: The path of the downloaded MP3 file.
    """
    from pytube import YouTube

    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=output_path, filename=filename)

    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return new_file
