# dependencies

# Installing ffmpeg
# choco install ffmpeg

# Installing pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Installing whisper
# pip install git+https://github.com/openai/whisper.git -q

# pip install streamlit

# pip install SpeechRecognition

# pip install pyaudio

import streamlit as st
import speech_recognition as sr
import whisper
import os
import pyaudio

st.title('Speech to Text App')

model = whisper.load_model("base")

# Create a temporary directory for uploaded audio
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

temp_audio_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_audio.wav"))

# Click on the button to speak and add the audio file to the sidebar
st.sidebar.header("Click to Speak")
record = st.sidebar.button("Record")
if record:
    # Record audio with the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    # Save the audio file to a temporary location
    with open(temp_audio_path, "wb") as f:
        f.write(audio.get_wav_data())
    st.sidebar.success("Recording Complete")
    st.sidebar.audio(temp_audio_path)

audio_exists = os.path.exists(temp_audio_path)
if st.sidebar.button("Transcribe Audio"):
    if audio_exists:    
        try:
            transcription = model.transcribe(temp_audio_path)
            st.sidebar.success("Transcription Complete")
            os.remove(temp_audio_path)
            st.markdown(transcription["text"])
        except Exception as e:
            st.sidebar.error(f"Error during transcription: {e}")
    else:
        st.sidebar.error("Please record an audio first")
