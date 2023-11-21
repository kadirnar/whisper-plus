import time

from whisperplus.pipelines.whisper import SpeechToTextPipeline
from whisperplus.utils.download_utils import download_and_convert_to_mp3


def main(url):
    video_path = download_and_convert_to_mp3(url)
    pipeline = SpeechToTextPipeline()
    start = time.time()
    transcript = pipeline(audio_path=video_path, model_id="openai/whisper-large-v3", language="turkish")
    end = time.time()
    print(transcript)
    print("Duration: ", end - start)
