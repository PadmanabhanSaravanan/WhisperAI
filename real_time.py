import streamlit as st
import speech_recognition as sr
import whisper
import pyaudio
import numpy as np
import openai

# Set up your OpenAI API key
openai.api_key = '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'


st.title('Real-time Speech to Text App')

model = whisper.load_model("base")

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

r = sr.Recognizer()

# Start recording button
start = st.button('Start Recording')
stop = st.button('Stop Recording')

transcriptions = ""

if start:
    st.write('Recording...')
    audio_data = []
    while not stop:
        # Continuously read from the audio stream
        buffer = stream.read(CHUNK_SIZE)
        audio_data.append(buffer)
        
        if len(audio_data) * CHUNK_SIZE / RATE > 5:  # transcribe every 5 seconds
            audio_chunk = b''.join(audio_data)
            audio = sr.AudioData(audio_chunk, RATE, 2)
            
            # Convert AudioData to numpy array
            audio_np = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32) / 32768
            
            try:
                transcription = whisper.transcribe(model=model, audio=audio_np, fp16=False)
                transcriptions += transcription["text"] + " "
                st.write(transcription["text"])
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            
            audio_data = []  # Clear buffer
if stop:
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]
    st.write("Recording stopped.")
    st.write("Transcriptions: " + transcriptions)
    prompt = transcriptions

    response = get_completion(prompt)
    st.write(response)

stream.stop_stream()
stream.close()
p.terminate()
