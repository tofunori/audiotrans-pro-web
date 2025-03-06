# AudioTrans Pro - Streamlit Version

A Streamlit-based version of AudioTrans Pro audio transcription application that can be deployed with just a few clicks. This application uses OpenAI's Whisper model via HuggingFace transformers to transcribe audio files to text.

## Features

- Transcribe audio files in multiple languages (English, French, German, Spanish, Italian)
- Options for precision and quality control
- Include timestamps in transcriptions
- Download results as TXT or DOCX files
- Modern and responsive web interface
- Easy deployment through Streamlit Sharing

## One-Click Deployment with Streamlit Cloud

Follow these steps to deploy AudioTrans Pro without any coding:

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account
3. Click on "New app"
4. Select your forked repository and branch (main)
5. Set the main file path to `app.py`
6. Click "Deploy"

Your app will be deployed automatically and you'll receive a URL that you can share with anyone!

## Usage

1. Load the model using the button in the sidebar
2. Upload an audio file (MP3, WAV, FLAC, OGG, or M4A)
3. Select language and other options
4. Click "Start Transcription"
5. Once complete, download the transcription as TXT or DOCX

## Notes on Performance

- The first load of the model may take a few minutes
- For better transcription quality, make sure audio files are clear
- Processing time depends on the file length and the selected quality
- GPU acceleration (if available on Streamlit Cloud) will significantly improve performance

## Quick Start - Local Development

1. Clone the repository:

```bash
git clone https://github.com/tofunori/audiotrans-pro-web.git
cd audiotrans-pro-web
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open your browser at http://localhost:8501

## License

This project is licensed under the MIT License.

## Acknowledgements

- Based on the original AudioTrans Pro desktop application
- Uses the Whisper model for speech recognition
- Built with Streamlit and HuggingFace Transformers