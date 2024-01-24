import logging
import tempfile

import ffmpeg
import torch
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
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

    def create_audio(self, video_path):
        audiofilename = video_path.replace(".mp4", '.mp3')
        input_stream = ffmpeg.input(video_path)
        audio = input_stream.audio
        output_stream = ffmpeg.output(audio, audiofilename)
        output_stream = ffmpeg.overwrite_output(output_stream)
        ffmpeg.run(output_stream)
        return audiofilename

    def add_subtitles_to_video(self, video_path, word_level_info, output_path):
        video = VideoFileClip(video_path)
        subtitles_clips = []

        padding = 20  # Sol ve sağ kenardan pixel cinsinden boşluk
        max_width = video.size[0] - 2 * padding  # Maksimum genişlik

        for chunk in word_level_info:
            text = chunk['text']
            start_time = chunk['timestamp'][0]
            end_time = chunk['timestamp'][1]

            # Altyazının genişliğini ve konumunu ayarla
            txt_clip = TextClip(text, fontsize=24, color='white', bg_color='black', size=(max_width, None))
            txt_clip = txt_clip.set_position(
                ('center', 'bottom')).set_start(start_time).set_duration(end_time - start_time)
            subtitles_clips.append(txt_clip)

        final_video = CompositeVideoClip([video, *subtitles_clips])
        final_video.write_videofile(output_path, codec="libx264", audio_codec='aac')
        return output_path

    def __call__(self, video_path: str, output_path: str, language: str = "turkish"):
        audio_path = self.create_audio(video_path)
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
        return self.add_subtitles_to_video(video_path, result['chunks'], output_path)
