# main.py
from audio_register import record_audio
from speech_to_text import transcribe_audio_file
import asyncio
from ollama import AsyncClient

systemContent = """
Sei un assistente virtuale specializzato nel riassumere testi lunghi in modo chiaro e informativo.
Leggi attentamente il testo fornito e genera un riassunto che includa tutte le informazioni principali e i dettagli rilevanti, privilegiando dati, fatti, concetti e spiegazioni chiave.
Ometti dettagli personali, aneddoti, opinioni soggettive, ringraziamenti e informazioni non essenziali.
Assicurati che il riassunto sia facilmente comprensibile anche da chi non conosce l’argomento, mantenendo un buon equilibrio tra completezza e sintesi.
Se il testo è tecnico, mantieni i dati e le spiegazioni più importanti; se è narrativo, concentrati sugli eventi e sui punti salienti.
Non limitare la lunghezza del riassunto, ma evita di essere prolisso.
"""

def main():
    # 1. Registra l'audio
    audio_file = record_audio()
    if not audio_file:
        print("Registrazione audio fallita.")
        return

    # 2. Trascrivi l'audio in testo
    testo = transcribe_audio_file(audio_file)
    if not testo:
        print("Trascrizione fallita.")
        return

    # 3. Riassumi il testo tramite AI
    async def chat():
        messages = [
            {'role': 'system', 'content': systemContent},
            {'role': 'user', 'content': 'riassumi il seguente testo: ' + testo}
        ]
        async for part in await AsyncClient().chat(model='gemma3:27b', messages=messages, stream=True):
            print(part['message']['content'], end='', flush=True)

    asyncio.run(chat())

if __name__ == "__main__":
    main()
