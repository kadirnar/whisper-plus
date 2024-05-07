import logging
from pathlib import Path

import yt_dlp

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def download_youtube_to_mp3(url, output_dir='./', filename="test"):
    """
    Downloads a YouTube video as an MP3 file.

    Parameters:
    url (str): The URL of the YouTube video to download.
    output_dir (str, optional): The directory to save the MP3 file. Defaults to the current working directory.
    filename (str, optional): The filename for the MP3 file. If not provided, the video title will be used.

    Returns:
    pathlib.Path: The path to the downloaded MP3 file.
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output filename
    if filename is None:
        filename = "%(title)s.%(ext)s"
    else:
        filename = f"{filename}.%(ext)s"

    # Download the video using yt_dlp
    ydl_opts = {
        'outtmpl': str(output_dir / filename),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Error downloading video: {e}")
        return None

    output_path = output_dir / filename.replace('%(ext)s', 'mp3')
    logging.info(f"Downloaded {output_path} as MP3")
    return str(output_path)


def download_youtube_to_mp4(url, output_dir='./', filename="test"):
    """
    Downloads a YouTube video as an MP4 file.

    Parameters:
    url (str): The URL of the YouTube video to download.
    output_dir (str, optional): The directory to save the MP4 file. Defaults to the current working directory.
    filename (str, optional): The filename for the MP4 file. If not provided, the video title will be used.

    Returns:
    str: The path to the downloaded MP4 file.
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output filename
    if filename is None:
        filename = "%(title)s.%(ext)s"
    else:
        filename = f"{filename}.%(ext)s"

    # Download the video using yt_dlp
    ydl_opts = {
        'outtmpl': str(output_dir / filename),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Error downloading video: {e}")
        return None

    output_path = output_dir / filename.replace('%(ext)s', 'mp4')
    logging.info(f"Downloaded {output_path} as MP4")
    return str(output_path)
