# Importing necessary modules required 
import speech_recognition as sr 
from googletrans import Translator 
from gtts import gTTS 
import os

# Capture Voice
# takes command through microphone
def takecommand(filepath):
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language='id')
    except Exception as e:
        print("Audio tidak terbaca")
        return "None"
    return query

# Taking voice input from the user