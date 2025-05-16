# audio_register.py
import sounddevice as sd
import numpy as np
import wave
import threading
import os

def record_audio(filename='output.wav', sample_rate=44100, channels=2):
    # Controlla se il file esiste già
    if os.path.exists(filename):
        scelta = input(
            f"\nEsiste già una registrazione salvata come '{filename}'.\n"
            "Vuoi registrare una nuova traccia? (s/n): "
        )
        if scelta.lower() != 's':
            print(f"Utilizzo la traccia già registrata: {filename}")
            return filename

    audio_chunks = []
    recording = True

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_int16 = (indata * 32767).astype(np.int16)
        audio_chunks.append(audio_int16.copy())

    def stop():
        input("Premi INVIO per fermare la registrazione...\n")
        nonlocal recording
        recording = False

    stop_thread = threading.Thread(target=stop, daemon=True)
    stop_thread.start()
    print("Registrazione avviata...")

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=np.float32, callback=callback):
        while recording:
            sd.sleep(100)
    print("Registrazione terminata.")

    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio salvato come {filename}")
        return filename
    else:
        print("Nessun audio è stato registrato")
        return None
