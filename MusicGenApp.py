import streamlit as st
from audiocraft.models import MusicGen
import torch
import torchaudio
import numpy as np
import base64
import os

token = st.secrets["huggingface_token"]

def get_one_direction_style():
    style_features = {
        'tempo': 'upbeat',
        'melody': 'catchy',
        'harmony': 'rich',
        'instruments': ['guitar', 'drums', 'keyboard'],
        'mood': 'joyful'
    }
    # This function returns a style description based on One Direction's typical music characteristics.
    return f"{style_features['tempo']} tempo, {style_features['melody']} melodies, {style_features['harmony']} harmonies, featuring {', '.join(style_features['instruments'])}, overall a {style_features['mood']} mood."


@st.cache_data(allow_output_mutation=True)
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def main():
    st.title("Music Generation Inspired by One Direction")
    st.write("Enter your generated lyrics and we will produce a track in the style of One Direction.")

    # Load the pre-trained model
    model = load_model()

    # Input for lyrics
    user_lyrics = st.text_area("Enter your lyrics here", "Type or paste your generated lyrics.")

    # Get One Direction style features
    style_prompt = get_one_direction_style()
    # Define genres
    genres = ["Rock", "Pop", "Melody"]
    genre = st.selectbox("Choose a genre:", genres)
    # Button to generate music
    if st.button("Generate Music"):
        # The generate_music function needs to accept a style description or parameters
        audio_tensor = model.generate_music(genre,lyrics=user_lyrics, style_description=style_prompt)

        # Convert tensor to audio file
        output_file = 'output.wav'
        torchaudio.save(output_file, audio_tensor, 44100)  # Assuming 44100 Hz sample rate

        # Play the generated music
        audio_file = open(output_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

if __name__ == "__main__":
    main()
