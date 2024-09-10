import streamlit as st 
import tempfile 
import os 
import google.generativeai as genai 
import librosa
import matplotlib.pyplot as plt
from dotenv import load_dotenv 
load_dotenv() 


## Configure Google API for audio summarization 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key = GOOGLE_API_KEY) 


# Sidebar for navigation and theme settings
st.sidebar.title("Audio Summarization App")
st.sidebar.markdown("Select options:")
theme = st.sidebar.radio("Select Theme:", ('Light', 'Dark'))
st.sidebar.markdown("## Recent Summaries")
# Add history (currently a placeholder)
for i in range(3):
    st.sidebar.text(f"Audio {i + 1}")

if theme == 'Dark':
    st.markdown('<style>body{background-color: #0e1117; color: white;}</style>', unsafe_allow_html=True)

def summarize_audio(audio_file_path):
    """Summarize the audio using Google's Generative API."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest") 
    audio_file = genai.upload_file(path=audio_file_path) 
    response = model.generate_content(["Please summarize the following audio.", audio_file])
    return response.text

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None


# Visualize audio waveform
def display_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y)
    ax.set(xlabel='Time (samples)', ylabel='Amplitude', title='Audio Waveform')
    st.pyplot(fig)


## Streamlit app interface 
st.title("Audio Summarization App")
st.write("## Upload an audio file (WAV/MP3) and get a summary!")

with st.expander("About this App"):
    st.write("""
    This app uses Google's generative AI to summarize audio files. 
    Upload your audio file in WAV or MP3 format and get a concise summary of its content. 
    You can also visualize the audio waveform and download the summary.
    """)

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    st.audio(audio_path)
    
    # Display audio file size and duration
    st.write(f"**File Size:** {audio_file.size / 1024:.2f} KB")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(f"**Duration:** {duration:.2f} seconds")
    
    # Display waveform
    st.write("### Audio Waveform")
    display_waveform(audio_path)

    if st.button('Summarize Audio'):
        with st.spinner('Summarizing...'):
            summary_text = summarize_audio(audio_path)
            st.info(summary_text)

        # Download option
        st.download_button(label="Download Summary", data=summary_text, file_name="audio_summary.txt", mime="text/plain")