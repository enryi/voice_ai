import speech_recognition as sr
import logging
from pydub import AudioSegment
import math
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def split_audio(file_path, chunk_length_ms=30000):
    """Split audio file into chunks"""
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    total_chunks = math.ceil(len(audio) / chunk_length_ms)
    logger.debug(f"Splitting audio into {total_chunks} chunks")
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    return chunks

def transcribe_audio_file(file_path):
    r = sr.Recognizer()
    full_text = []
    chunk_files = []  # Lista per memorizzare i chunk temporanei
    
    try:
        logger.debug(f"Processing audio file: {file_path}")
        chunks = split_audio(file_path)
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            # Export chunk to temporary file
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_files.append(chunk_path)  # Aggiungi il percorso alla lista
            
            with sr.AudioFile(chunk_path) as source:
                audio = r.record(source)
                try:
                    text = r.recognize_google(audio, language="it-IT")
                    full_text.append(text)
                    logger.debug(f"Chunk {i+1} transcribed successfully")
                except sr.UnknownValueError:
                    logger.warning(f"Could not understand audio in chunk {i+1}")
                except sr.RequestError as e:
                    logger.error(f"API error in chunk {i+1}: {str(e)}")
        
        final_text = " ".join(full_text)
        print("\nContenuto del file:")
        print(final_text)

        # Eliminazione chunk temporanei
        logger.debug("\nPulizia file temporanei...")
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
                logger.debug(f"Eliminato: {chunk_file}")
            except Exception as e:
                logger.error(f"Errore eliminazione {chunk_file}: {str(e)}")

        # Richiesta eliminazione file originale
        user_choice = input("\nVuoi eliminare il file audio originale? (sì/no): ").strip().lower()
        if user_choice in ['sì', 'si', 'yes', 'y']:
            try:
                os.remove(file_path)
                logger.debug(f"File originale eliminato: {file_path}")
                print("File audio originale eliminato con successo")
            except Exception as e:
                logger.error(f"Errore eliminazione file originale: {str(e)}")
                print("Errore nell'eliminazione del file originale")
        else:
            print("File originale conservato")

        return final_text
    
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__} - {str(e)}")
        print(f"Errore durante la lettura del file: {str(e)}")
        return ""
