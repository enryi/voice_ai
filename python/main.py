from audio_register import record_audio
from speech_to_text import transcribe_audio_file
import asyncio
import os
from ollama import AsyncClient
from typing import Optional
import logging
import time
from datetime import timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Sei un assistente virtuale esperto nel riassumere testi lunghi in modo chiaro, accurato e informativo.

1. Leggi attentamente il testo fornito e individua tutte le informazioni principali, i dati chiave, i concetti e le spiegazioni essenziali.

2. Il riassunto deve essere proporzionato alla lunghezza e alla densità informativa del testo originale: per un testo molto lungo e ricco di contenuti, il riassunto deve essere articolato e dettagliato, coprendo tutti i temi e gli argomenti trattati.

3. Non omettere parti rilevanti: includi spiegazioni, dati, riferimenti a opere, argomentazioni, esempi e riflessioni che risultino centrali per la comprensione del testo.

4. Ometti solo dettagli personali, aneddoti privati, opinioni soggettive e ringraziamenti, ma mantieni tutti i contenuti che contribuiscono alla sostanza del discorso o all’esposizione dei concetti.

5. Mantieni un equilibrio tra completezza e sintesi: il riassunto deve essere esaustivo ma non prolisso, organizzato in modo logico e suddiviso in paragrafi o punti tematici se necessario.

6. Se il testo è tecnico, spiega i dati e i termini più importanti; se è narrativo, riporta tutti gli eventi e le riflessioni principali, non solo i macro-temi.

7. Evita ripetizioni, ma assicurati che nessun tema o argomento centrale venga trascurato.

8. Non limitare la lunghezza del riassunto: il riassunto deve essere sufficientemente lungo da coprire tutti gli aspetti rilevanti del testo originale, senza ridurlo eccessivamente.

9. Concludi con una frase che sintetizzi l'importanza, l'impatto o lo stato attuale dell'argomento.
"""

def split_text_into_chunks(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks while preserving sentence structure."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        if current_length + len(para) > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(para)
        current_length += len(para) + 2
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

async def summarize_chunk(chunk_file: str, chunk_num: int, total_chunks: int) -> str:
    """Summarize a single chunk of text."""
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunk_text = f.read()
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Questo è il chunk {chunk_num}/{total_chunks}. '
                                  f'Raccogli le informazioni chiave: {chunk_text}'}
    ]
    
    try:
        print(f"\nProcessing chunk {chunk_num}/{total_chunks}:")
        print("-" * 50)
        
        summary = []
        async for part in await AsyncClient().chat(
            model='llama3.2:latest',
            messages=messages,
            stream=True
        ):
            content = part['message']['content']
            summary.append(content)
            print(content, end='', flush=True)
        summary_text = ''.join(summary)
        summary_file = f"text/summary_{chunk_num:03d}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        return summary_text
    except Exception as e:
        logger.error(f"Summarization error for chunk {chunk_num}: {e}")
        return ""

async def summarize_text(text: str) -> None:
    """Generate summary using Ollama API, processing text chunks sequentially."""
    try:
        with open("text/chunks_metadata.json", 'r') as f:
            metadata = json.load(f)
        summaries = []
        for i, chunk_file in enumerate(metadata['chunk_files'], 1):
            summary = await summarize_chunk(chunk_file, i, metadata['total_chunks'])
            summaries.append(summary)
        print("\n\nFinal Combined Summary:")
        print("=" * 80)
        print('\n\n'.join(summaries))
        
        with open("text/final_summary.txt", 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(summaries))
        print("\nCleaning up temporary files...")
        try:
            for chunk_file in metadata['chunk_files']:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            os.remove("text/chunks_metadata.json")
            keep_files = {'final_summary.txt', 'refined_summary.txt'}
            for file in os.listdir("text"):
                if file not in keep_files:
                    file_path = os.path.join("text", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
            
    except Exception as e:
        logger.error(f"Error in summarization process: {e}")

async def refine_final_summary(summary_file: str = "text/final_summary.txt") -> None:
    """Refine the final summary by processing it in smaller chunks."""
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = split_text_into_chunks(text)
        print(f"\nRefining summary in {len(chunks)} chunks...")
        
        refined_chunks = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}:")
            print("=" * 50)
            
            messages = [
                {'role': 'system', 'content': """Sei un editor esperto. Il tuo compito è:
1. Rimuovere tutte le ripetizioni di concetti e informazioni
2. Eliminare frasi incomplete o poco chiare
3. Mantenere la coerenza tra i paragrafi
4. Preservare tutte le informazioni uniche e rilevanti
5. Migliorare la leggibilità del testo"""},
                {'role': 'user', 'content': f'Riorganizza e pulisci questo testo:\n\n{chunk}'}
            ]
            
            refined_chunk = []
            async for part in await AsyncClient().chat(
                model='llama3.2:latest',
                messages=messages,
                stream=True
            ):
                content = part['message']['content']
                refined_chunk.append(content)
                print(content, end='', flush=True)
            
            refined_chunks.append(''.join(refined_chunk))
        final_refined_text = '\n\n'.join(refined_chunks)
        with open("text/refined_summary.txt", 'w', encoding='utf-8') as f:
            f.write(final_refined_text)
        
        logger.info("Summary refinement completed")
        print("\nCleaning up temporary files...")
        try:
            if os.path.exists(summary_file):
                os.remove(summary_file)
            
            for file in os.listdir("text"):
                if file != "refined_summary.txt":
                    file_path = os.path.join("text", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            logger.info("All temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
        
    except Exception as e:
        logger.error(f"Error refining summary: {e}")

async def main() -> None:
    """Main application flow with error handling and step timing."""
    total_start = time.time()
    
    try:
        record_start = time.time()
        if not (audio_file := record_audio()):
            logger.error("Audio recording failed")
            return
        record_time = time.time() - record_start
        
        transcribe_start = time.time()
        if not await transcribe_audio_file(audio_file):
            logger.error("Transcription failed")
            return
        transcribe_time = time.time() - transcribe_start

        summarize_start = time.time()
        await summarize_text("")
        await refine_final_summary()
        summarize_time = time.time() - summarize_start
        
        total_time = time.time() - total_start
        
        print("\n\nTiming Statistics:")
        print(f"Recording time: {str(timedelta(seconds=int(record_time)))}")
        print(f"Transcription time: {str(timedelta(seconds=int(transcribe_time)))}")
        print(f"Summarization time: {str(timedelta(seconds=int(summarize_time)))}")
        print(f"Total execution time: {str(timedelta(seconds=int(total_time)))}")

    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())