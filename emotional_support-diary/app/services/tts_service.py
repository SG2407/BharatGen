from TTS.api import TTS
import io

class TTSService:
    def __init__(self):
        # Load the TTS model (e.g., Coqui TTS pre-trained model)
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)

    def generate_audio_response(self, text: str) -> io.BytesIO:
        # Generate audio from text
        audio_path = "response.wav"
        self.tts.tts_to_file(text=text, file_path=audio_path)
        
        # Load the audio into a BytesIO object for streaming
        audio_buffer = io.BytesIO()
        with open(audio_path, "rb") as f:
            audio_buffer.write(f.read())
        audio_buffer.seek(0)
        return audio_buffer
