from audio_register import record_audio
from speech_to_text import transcribe_audio_file
import asyncio
from ollama import AsyncClient
from typing import Optional
import logging
import time
from datetime import timedelta

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

9. Concludi con una frase che sintetizzi l’importanza, l’impatto o lo stato attuale dell’argomento.
"""

async def summarize_text(text: str) -> None:
    """Generate summary using Ollama API."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'riassumi il seguente testo: {text}'}
    ]
    
    try:
        async for part in await AsyncClient().chat(
            model='llama3.2:latest', 
            messages=messages, 
            stream=True
        ):
            print(part['message']['content'], end='', flush=True)
    except Exception as e:
        logger.error(f"Summarization error: {e}")

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
        '''
        if not (transcribed_text := transcribe_audio_file(audio_file)):
            logger.error("Transcription failed")
            return
        '''
        transcribe_time = time.time() - transcribe_start
        text_file = "audio/testo.txt"
        summarize_start = time.time()
        with open(text_file, 'r', encoding='utf-8') as f:
                transcribed_text = f.read()
        await summarize_text(transcribed_text)
        summarize_time = time.time() - summarize_start
        
        total_time = time.time() - total_start
        
        print("\n\nTiming Statistics:")
        print(f"Recording time: {str(timedelta(seconds=int(record_time)))}")
        print(f"Transcription time: {str(timedelta(seconds=int(transcribe_time)))}")
        print(f"Summarization time: {str(timedelta(seconds=int(summarize_time)))}")
        print(f"Total execution time: {str(timedelta(seconds=int(total_time)))}")
        
        logger.info(f"Pipeline completed in {str(timedelta(seconds=int(total_time)))}")

    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())