import speech_recognition as sr
import logging
from pydub import AudioSegment
from typing import List, Optional, Dict
import os
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_audio(file_path: str, chunk_length_ms: int = 30000) -> List[AudioSegment]:
    """Split audio file into manageable chunks."""
    try:
        audio = AudioSegment.from_wav(file_path)
        return [audio[i:i + chunk_length_ms] 
                for i in range(0, len(audio), chunk_length_ms)]
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return []

def process_chunk(chunk_data: Dict) -> Optional[str]:
    """Process individual audio chunk with multiprocessing support."""
    chunk = chunk_data['chunk']
    chunk_num = chunk_data['chunk_num']
    total_chunks = chunk_data['total_chunks']
    
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        processed_chunk = (chunk
                         .set_frame_rate(16000)
                         .set_channels(1))
        audio_data = sr.AudioData(
            processed_chunk.raw_data,
            sample_rate=16000,
            sample_width=2
        )
        text = recognizer.recognize_google(audio_data, language="it-IT")
        logger.info(f"Chunk {chunk_num}/{total_chunks} transcribed successfully")
        return text
    except sr.UnknownValueError:
        logger.warning(f"Chunk {chunk_num}: Speech not understood")
    except sr.RequestError as e:
        logger.error(f"Chunk {chunk_num}: API error - {str(e)}")
    except Exception as e:
        logger.error(f"Chunk {chunk_num}: Unexpected error - {str(e)}")
    return None

def transcribe_audio_file(file_path: str) -> str:
    """Main transcription function using multiprocessing."""
    if not os.path.exists(file_path):
        logger.error("File not found")
        return ""

    try:
        chunks = split_audio(file_path)
        if not chunks:
            return ""

        total_chunks = len(chunks)
        chunk_data_list = [
            {
                'chunk': chunk,
                'chunk_num': i + 1,
                'total_chunks': total_chunks
            }
            for i, chunk in enumerate(chunks)
        ]

        # Use maximum number of cores minus 1 to keep system responsive
        num_processes = max(1, cpu_count() - 1)
        logger.info(f"Processing with {num_processes} cores")
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_chunk, chunk_data_list)
            
        # Filter out None results and join texts
        final_text = " ".join(filter(None, results))
        
        print("\nFile content:")
        print(final_text)

        if input("\nDelete original audio file? (y/n): ").lower().strip() in {'y', 'yes'}:
            try:
                os.remove(file_path)
                logger.info("Original file deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting file: {str(e)}")

        return final_text

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return ""

if __name__ == '__main__':
    transcribe_audio_file("audio/input.wav")