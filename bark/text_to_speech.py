import logging

from scipy.io.wavfile import write

from bark import SAMPLE_RATE, generate_audio, preload_models


class TextToSpeechPipeline:

    def __init__(self, input_text: str):
        self.input_text = input_text

        if not self.input_text:
            logging.warning("Input text is empty. Cannot convert to speech.")
            # You may raise an exception or return a specific value if needed

    def convert_to_speech(self):
        if not self.input_text:
            logging.warning("No text to convert to speech. Aborting.")
            return

        audio_array = generate_audio(self.input_text)
        write("output2.wav", rate=SAMPLE_RATE, data=audio_array)
