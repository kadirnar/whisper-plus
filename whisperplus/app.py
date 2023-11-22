import gradio as gr

from whisperplus.pipelines.whisper import SpeechToTextPipeline
from whisperplus.utils.download_utils import download_and_convert_to_mp3


def main(url, language_choice):
    video_path = download_and_convert_to_mp3(url)
    pipeline = SpeechToTextPipeline()
    transcript = pipeline(audio_path=video_path, model_id="openai/whisper-large-v3", language=language_choice)
    words_to_delete = ["Altyazımiyorum", "Altyazı M.K"]

    for word in words_to_delete:
        transcript = transcript.replace(word, '')

    return transcript, video_path


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                youtube_url_path = gr.Text(placeholder="Enter Youtube URL", label="Youtube URL")

                language_choice = gr.Dropdown(
                    choices=[
                        "english",
                        "turkish",
                        "german",
                        "french",
                        "chinese",
                        "japanese",
                        "korean",
                    ],
                    value="turkish",
                    label="Language",
                )

                whisperplus_in_predict = gr.Button(value="Generator")

            with gr.Column():
                output_text = gr.Textbox(placeholder="Output Text")
                output_audio = gr.Audio(label="Output Audio")

        whisperplus_in_predict.click(
            fn=main,
            inputs=[
                youtube_url_path,
                language_choice,
            ],
            outputs=[output_text, output_audio],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    WhisperPlus: Advancing Speech-to-Text Processing 🚀
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Follow me for more!
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch()
