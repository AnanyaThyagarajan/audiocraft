import streamlit as st
from audiocraft.models import MusicGen
import torch
import torchaudio
import numpy as np
import base64
import os


@st.cache(allow_output_mutation=True)
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def main():
    st.title("Music Generation Inspired by One Direction")
    st.write("Select a music genre and generate a track similar to One Direction's style.")

    # Load the pre-trained model
    model = load_model()

    # Define genres
    genres = ["Rock", "Pop", "Melody"]
    genre = st.selectbox("Choose a genre:", genres)

    # Button to generate music
    if st.button("Generate Music"):
        # Placeholder for generating music based on the selected genre
        # This assumes the model accepts a genre and outputs an audio tensor
        audio_tensor = model.generate_music(genre)  # This method needs to be defined or adjusted according to your model's API

        # Convert tensor to audio file
        output_file = 'output.wav'
        torchaudio.save(output_file, audio_tensor, 44100)  # Assuming 44100 Hz sample rate

        # Convert audio file to base64 to embed in HTML
        audio_file = open(output_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

if __name__ == "__main__":
    main()
