﻿# EchoBrief

**EchoBrief** is an AI-powered audio summarization application that simplifies the process of extracting key information from audio files. Whether it's a podcast, lecture, or meeting recording, EchoBrief uses Google's Generative AI to provide accurate and concise summaries.

## Features
- **Audio Summarization**: Upload your WAV or MP3 files and receive a concise summary using Google's Generative AI.
- **Waveform Visualization**: Visualize the waveform of the uploaded audio file for better insights.
- **File Info**: View audio file details such as file size and duration.
- **Download Summary**: Easily download the generated summary in a text format.
- **Dark Mode**: Switch between light and dark themes for a personalized experience.
- **Recent History**: View the list of recently summarized audio files.

## Technologies Used
- **Python**
- **Streamlit**: Web framework for building interactive UI.
- **Google Generative AI**: AI model for summarizing content.
- **Librosa**: Library for audio analysis and visualization.
- **Matplotlib**: Visualization of the audio waveform.
- **dotenv**: Manage environment variables.
  

## Getting Started

### Prerequisites
To run this project, you’ll need:
- **Python 3.7+**
- **Google API Key**: You need to have a Google API key for the Generative AI.
- **pip**: Python package manager.

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/EchoBrief.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd EchoBrief
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your Google API key**:
    - Create a `.env` file in the root of the project.
    - Add your Google API key in the `.env` file:
      ```bash
      GOOGLE_API_KEY=your_api_key
      ```

### Running the App

To start the Streamlit app, run the following command:

```bash
streamlit run app.py
