import logging
import tempfile

import srt
import torch
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WhisperAutoCaptionPipeline:

    def __init__(self, model_id: str = "openai/whisper-large-v3"):
        self.model = None
        self.device = None

        if self.model is None:
            self.load_model(model_id)
        else:
            logging.info("Model already loaded.")

        self.set_device()

    def set_device(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, model_id: str = "openai/whisper-large-v3"):
        logging.info("Loading model...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        logging.info("Model loaded successfully.")

    def __call__(
            self,
            video_path: str,
            audio_path: str,
            language: str = "turkish",
            output_path: str = "output.mp4"):
        logging.info("Transcribing audio...")
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float16,
            chunk_length_s=30,
            max_new_tokens=128,
            batch_size=24,
            return_timestamps=True,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            generate_kwargs={"language": language},
        )

        result = pipe(audio_path)

        word_level_info = []
        for chunk in result['chunks']:
            words = chunk['text'].split()
            start_time = chunk['timestamp'][0]
            end_time = chunk['timestamp'][1]

            if len(words) > 1:
                duration_per_word = (end_time - start_time) / len(words)
                times = [(start_time + i * duration_per_word, start_time + (i + 1) * duration_per_word)
                         for i in range(len(words))]
            else:
                times = [(start_time, end_time)]

            for word, (start, end) in zip(words, times):
                word_level_info.append({'word': word, 'start': start, 'end': end})

        logging.info("Adding subtitles to video...")
        return self.add_subtitles_to_video(video_path, word_level_info, output_path)

    def generate_subtitles(self, word_level_info, ahead_time=3):
        subtitles = []
        start = 0
        while start < word_level_info[-1]['end']:
            end = min(start + ahead_time, word_level_info[-1]['end'])
            text = ' '.join(
                [word_info['word'] for word_info in word_level_info if start <= word_info['start'] < end])
            subtitles.append(
                srt.Subtitle(
                    index=len(subtitles),
                    start=srt.timedelta(seconds=start),
                    end=srt.timedelta(seconds=end),
                    content=text))
            start += 1
        return srt.compose(subtitles)

    def add_subtitles_to_video(self, video_path, word_level_info, output_path=None):
        video = VideoFileClip(video_path)
        subtitles_clips = []

        segment_duration = 6
        for start_time in range(0, int(video.duration), segment_duration):
            end_time = min(start_time + segment_duration, video.duration)

            segment_text = ' '.join([
                word_info['word'] for word_info in word_level_info
                if start_time <= word_info['start'] < end_time
            ])

            if segment_text:
                txt_clip = TextClip(segment_text, fontsize=30, color='yellow', bg_color='black')
                txt_clip = txt_clip.set_position('bottom').set_start(start_time).set_duration(
                    end_time - start_time)
                subtitles_clips.append(txt_clip)

        final_video = CompositeVideoClip([video, *subtitles_clips])
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')
        final_video.write_videofile(output_path, codec="libx264", audio_codec='aac')
        return output_path
