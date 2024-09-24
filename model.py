from flask import Flask, request, render_template
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from pydub import AudioSegment
import os
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'  # Directory to save uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Function to load and preprocess audio
def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Mono and 16kHz sampling rate
    audio_tensor = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)
    return audio_tensor

# Function to transcribe the audio using Wav2Vec 2.0
def transcribe_audio(input_audio):
    input_values = processor(input_audio, return_tensors="pt", padding=True).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Function to fetch latitude and longitude from the user's IP
def fetch_lat_lon_from_ip():
    try:
        # Example using ipinfo.io
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data.get("loc")  # Location in the format "lat,lon"
        if loc:
            lat, lon = loc.split(",")
            return lat, lon
    except Exception as e:
        print(f"Error fetching location: {e}")
    return None, None

# Function to detect human voice based on transcription
def detect_human_voice(transcription):
    if transcription.strip():  # If the transcription contains words
        return "Human voice detected.", fetch_lat_lon_from_ip()  # Fetch latitude and longitude
    else:
        return "No human voice detected.", (None, None)

# Route for uploading files
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load and preprocess the audio
            input_audio = load_audio(file_path)

            # Transcribe the audio
            transcription = transcribe_audio(input_audio)

            # Detect if human voice is present and fetch location
            voice_detection, (lat, lon) = detect_human_voice(transcription)

            return render_template('result.html', transcription=transcription,
                                   voice_detection=voice_detection, latitude=lat, longitude=lon)

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
