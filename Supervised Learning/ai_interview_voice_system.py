import os
import speech_recognition as sr
import pyttsx3
import torch
import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import librosa
import tensorflow as tf



class AIInterviewVoiceSystem:
    def __init__(self):
        # Initialize Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()
        
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        
        # Load Pre-trained Transformer Model
        self.speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Initialize Emotion Detection Model
        self.emotion_model = self._load_emotion_model()
    
    def _load_emotion_model(self):
        """
        Load a simple emotion detection model (placeholder).
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),  # Input: Feature vector
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')  # Output: 5 emotion classes
        ])
        print("Emotion detection model initialized (weights not preloaded).")
        return model
    
    def speak_question(self, question):
        """
        Convert text to speech using pyttsx3.
        """
        try:
            self.tts_engine.setProperty('rate', 150)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.9)  # Volume level
            self.tts_engine.say(question)
            self.tts_engine.runAndWait()
            print("Spoken Question:", question)
            return True
        except Exception as e:
            print(f"Error in TTS: {e}")
            return False
    
    def record_answer(self, duration=10):
        """
        Record audio using sounddevice.
        """
        print("Recording started. Please speak your answer...")
        fs = 44100  # Sampling rate
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait for the recording to finish
            output_file = "candidate_answer.wav"
            sf.write(output_file, recording, fs)
            print(f"Recording saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error during recording: {e}")
            return None
    
    def transcribe_answer(self, audio_file):
        """
        Transcribe audio using SpeechRecognition and Wav2Vec2.
        """
        transcripts = {}
        
        # Method 1: SpeechRecognition
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                transcripts['google'] = self.recognizer.recognize_google(audio)
                transcripts['sphinx'] = self.recognizer.recognize_sphinx(audio)
        except Exception as e:
            print(f"Error in SpeechRecognition: {e}")
        
        # Method 2: Wav2Vec2
        try:
            input_audio, sample_rate = librosa.load(audio_file, sr=16000)
            input_values = self.speech_processor(input_audio, sampling_rate=sample_rate, return_tensors="pt").input_values
            logits = self.speech_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcripts['transformer'] = self.speech_processor.batch_decode(predicted_ids)[0]
        except Exception as e:
            print(f"Error in Wav2Vec2 Transcription: {e}")
        
        return transcripts
    
    def detect_emotion(self, audio_file):
        """
        Detect emotion using placeholder features and TensorFlow model.
        """
        try:
            features = self._extract_audio_features(audio_file)
            predictions = self.emotion_model.predict(features)
            emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised']
            detected_emotion = emotions[np.argmax(predictions)]
            return detected_emotion
        except Exception as e:
            print(f"Error in Emotion Detection: {e}")
            return None
    
    def _extract_audio_features(self, audio_file):
        """
        Extract MFCC features for emotion detection.
        """
        y, sr = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        return features.reshape(1, -1)

# Test Example
def test_system():
    system = AIInterviewVoiceSystem()
    
    # Test TTS
    system.speak_question("What motivates you to apply for this job?")
    
    # Test Recording
    audio_file = system.record_answer(duration=5)  # Adjust duration as needed
    
    # Test Transcription
    if audio_file:
        transcripts = system.transcribe_answer(audio_file)
        print("\nTranscriptions:")
        for method, text in transcripts.items():
            print(f"{method}: {text}")
    
    # Test Emotion Detection
    if audio_file:
        emotion = system.detect_emotion(audio_file)
        print("\nDetected Emotion:", emotion)

# Run the test
if __name__ == "__main__":
    test_system()
