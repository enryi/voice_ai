import speech_recognition as sr
import logging
from pydub import AudioSegment
from typing import List, Optional, Dict, Tuple
import os
from multiprocessing import Pool, cpu_count
import json
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_audio(file_path: str, chunk_length_ms: int = 150000) -> List[AudioSegment]:
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

async def process_and_save_chunks(chunks: List[Dict], output_dir: str = "text") -> List[Tuple[int, str]]:
    """Process chunks and save them sequentially."""
    os.makedirs(output_dir, exist_ok=True)
    processed_chunks = []
    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Processing with {num_processes} cores")
    loop = asyncio.get_event_loop()
    
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap(process_chunk, chunks), 1):
            if result:
                chunk_file = os.path.join(output_dir, f"chunk_{i:03d}.txt")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(result)
                processed_chunks.append((i, chunk_file))
                logger.info(f"Saved chunk {i} to {chunk_file}")
    metadata = {
        "total_chunks": len(chunks),
        "processed_chunks": len(processed_chunks),
        "chunk_files": [f[1] for f in processed_chunks]
    }
    with open(os.path.join(output_dir, "chunks_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return processed_chunks

async def transcribe_audio_file(file_path: str) -> bool:
    """Main transcription function using multiprocessing."""
    if not os.path.exists(file_path):
        logger.error("File not found")
        return False

    try:
        chunks = split_audio(file_path)
        if not chunks:
            return False

        total_chunks = len(chunks)
        chunk_data_list = [
            {
                'chunk': chunk,
                'chunk_num': i + 1,
                'total_chunks': total_chunks
            }
            for i, chunk in enumerate(chunks)
        ]
        processed_chunks = await process_and_save_chunks(chunk_data_list)
        
        if input("\nDelete original audio file? (y/n): ").lower().strip() in {'y', 'yes'}:
            try:
                os.remove(file_path)
                logger.info("Original file deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting file: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return False

if __name__ == '__main__':
    asyncio.run(transcribe_audio_file("audio/input.wav"))