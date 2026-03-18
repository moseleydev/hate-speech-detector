python3 -m venv venv - Create the virtual environment (named 'venv')

source venv/bin/activate - start env

uvicorn server:app --reload
uvicorn hate_speech_api:app --reload

pip3 cache purge - removes caches downloads
