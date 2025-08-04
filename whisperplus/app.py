import gradio as gr
import torch
from transformers import BitsAndBytesConfig, HqqConfig

from whisperplus import (
    SpeechToTextPipeline,
    download_youtube_to_mp3,
    download_youtube_to_mp4,
    format_speech_to_dialogue,
)
from whisperplus.pipelines.long_text_summarization import LongTextSummarizationPipeline
from whisperplus.pipelines.summarization import TextSummarizationPipeline
from whisperplus.pipelines.text2speech import TextToSpeechPipeline
from whisperplus.pipelines.whisper_autocaption import WhisperAutoCaptionPipeline
from whisperplus.pipelines.whisper_diarize import ASRDiarizationPipeline


def youtube_url_to_text(url, model_id, language_choice):
    """
    Main function that downloads and converts a video to MP3 format, performs speech-to-text conversion using
    a specified model, and returns the transcript along with the video path.

    Args:
        url (str): The URL of the video to download and convert.
        model_id (str): The ID of the speech-to-text model to use.
        language_choice (str): The language choice for the speech-to-text conversion.

    Returns:
        transcript (str): The transcript of the speech-to-text conversion.
    """
    audio_path = download_youtube_to_mp3(url, output_dir="downloads", filename="test")

    hqq_config = HqqConfig(
        nbits=4,
        group_size=64,
        quant_zero=False,
        quant_scale=False,
        axis=0,
        offload_meta=False,
    )  # axis=0 is used by default

    pipeline = SpeechToTextPipeline(
        model_id=model_id,
        quant_config=hqq_config,
        flash_attention_2=True,
    )

    transcript = pipeline(
        audio_path=audio_path,
        chunk_length_s=30,
        stride_length_s=5,
        max_new_tokens=128,
        batch_size=100,
        language=language_choice,
        return_timestamps=False,
    )
    return transcript


def summarization(text, model_id="facebook/bart-large-cnn"):
    """
    Main function that performs summarization using a specified model and returns the summary.

    Args:
        text (str): The text to summarize.
        model_id (str): The ID of the summarization model to use.

    Returns:
        summary (str): The summary of the text.
    """
    summarizer = TextSummarizationPipeline(model_id=model_id)
    summary = summarizer.summarize(text)

    return summary[0]["summary_text"]


def long_text_summarization(text, model_id="facebook/bart-large-cnn"):
    """
    Main function that performs summarization using a specified model and returns the summary.

    Args:
        text (str): The text to summarize.
        model_id (str): The ID of the summarization model to use.

    Returns:
        summary (str): The summary of the text.
    """
    summarizer = LongTextSummarizationPipeline(model_id=model_id)
    summary_text = summarizer.summarize(text)

    return summary_text


def speaker_diarization(url, model_id, device, num_speakers, min_speaker, max_speaker):
    """
    Main function that downloads and converts a video to MP3 format, performs speech-to-text conversion using
    a specified model, and returns the transcript along with the video path.

    Args:
        url (str): The URL of the video to download and convert.
        model_id (str): The ID of the speech-to-text model to use.
        language_choice (str): The language choice for the speech-to-text conversion.

    Returns:
        transcript (str): The transcript of the speech-to-text conversion.
        video_path (str): The path of the downloaded video.
    """
    pipeline = ASRDiarizationPipeline.from_pretrained(
        asr_model=model_id,
        diarizer_model="pyannote/speaker-diarization",
        use_auth_token=False,
        chunk_length_s=30,
        device=device,
    )

    audio_path = download_youtube_to_mp3(url)
    output_text = pipeline(
        audio_path, num_speakers=num_speakers, min_speaker=min_speaker, max_speaker=max_speaker)
    dialogue = format_speech_to_dialogue(output_text)
    return dialogue, audio_path


def text2spech_bark(text, model_id="suno/bark", voice_preset="v2/en_speaker_6"):
    tts = TextToSpeechPipeline(model_id=model_id)
    audio = tts(text=text, voice_preset=voice_preset)
    return audio


def whisper_autocaption(url, language, model_id="openai/whisper-large-v3"):
    video_path = download_youtube_to_mp4(url)

    caption = WhisperAutoCaptionPipeline(model_id=model_id)
    output = caption(video_path=video_path, output_path="output.mp4", language=language)
    return output


with gr.Blocks() as demo:
    with gr.Tab("YouTube URL to Text"):
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="Enter YouTube URL")
                model_id_input = gr.Textbox(label="Enter Model ID", value="openai/whisper-medium")
                language_input = gr.Textbox(label="Enter Language", value="en")
                submit_btn1 = gr.Button("Submit")
            with gr.Column():
                output1 = gr.Textbox(label="Transcript")
        submit_btn1.click(
            youtube_url_to_text, inputs=[url_input, model_id_input, language_input], outputs=output1)

    with gr.Tab("Text Summarization"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Enter Text", lines=5)
                model_id_input2 = gr.Textbox(label="Enter Model ID", value="facebook/bart-large-cnn")
                submit_btn2 = gr.Button("Summarize")
            with gr.Column():
                output2 = gr.Textbox(label="Summary")
        submit_btn2.click(summarization, inputs=[text_input, model_id_input2], outputs=output2)

    with gr.Tab("Long Text Summarization"):
        with gr.Row():
            with gr.Column():
                long_text_input = gr.Textbox(label="Enter Long Text", lines=10)
                model_id_input3 = gr.Textbox(label="Enter Model ID", value="facebook/bart-large-cnn")
                submit_btn3 = gr.Button("Summarize Long Text")
            with gr.Column():
                output3 = gr.Textbox(label="Long Text Summary")
        submit_btn3.click(long_text_summarization, inputs=[long_text_input, model_id_input3], outputs=output3)

    with gr.Tab("Speaker Diarization"):
        with gr.Row():
            with gr.Column():
                url_input2 = gr.Textbox(label="Enter YouTube URL")
                model_id_input4 = gr.Textbox(label="Enter Model ID")
                num_speakers = gr.Number(label="Number of Speakers", value=2)
                min_speakers = gr.Number(label="Min Speakers", value=1)
                max_speakers = gr.Number(label="Max Speakers", value=4)
                device = gr.Textbox(label="Device", value="cpu")
                submit_btn4 = gr.Button("Diarize")
            with gr.Column():
                output4 = gr.DataFrame(headers=["Speaker", "Text"], datatype=["str", "str"])
        submit_btn4.click(
            speaker_diarization,
            inputs=[url_input2, model_id_input4, device, num_speakers, min_speakers, max_speakers],
            outputs=output4)

    with gr.Tab("Text to Speech"):
        with gr.Row():
            with gr.Column():
                text_input2 = gr.Textbox(label="Enter Text", lines=3)
                model_id_input5 = gr.Textbox(label="Enter Model ID", value="suno/bark")
                voice_preset = gr.Textbox(label="Voice Preset", value="v2/en_speaker_6")
                submit_btn5 = gr.Button("Generate Audio")
            with gr.Column():
                output5 = gr.Audio(label="Generated Audio")
        submit_btn5.click(
            text2spech_bark, inputs=[text_input2, model_id_input5, voice_preset], outputs=output5)

    with gr.Tab("Whisper Autocaption"):
        with gr.Row():
            with gr.Column():
                url_input3 = gr.Textbox(label="Enter YouTube URL")
                language = gr.Textbox(label="Language", value="en")
                model_id_input6 = gr.Textbox(label="Enter Model ID", value="openai/whisper-large-v2")
                submit_btn6 = gr.Button("Generate Captions")
            with gr.Column():
                output6 = gr.Video(label="Captioned Video")
        submit_btn6.click(
            whisper_autocaption, inputs=[url_input3, language, model_id_input6], outputs=output6)

demo.launch()
