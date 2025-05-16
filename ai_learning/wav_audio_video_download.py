import os
import yt_dlp
import re
import unicodedata

def sanitize_filename(filename):
    # Rimuovi l'estensione se presente
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    # Minuscolo e spazi in underscore
    name = name.lower().replace(' ', '_')
    # Normalizza unicode per togliere gli accenti
    name = unicodedata.normalize('NFD', name)
    name = name.encode('ascii', 'ignore').decode('utf-8')
    # Tieni solo lettere, numeri e underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    return f"{name}.wav"

def force_rename(src, dst):
    # Elimina il file di destinazione se esiste già (Windows)
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)

def rename_all_wav_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            sanitized = sanitize_filename(filename)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, sanitized)
            if old_path != new_path:
                force_rename(old_path, new_path)
                print(f"RINOMINATO: {filename} -> {sanitized}")

def download_and_convert(url, output_dir='ai_learning/audio'):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            
            # Rinomina DOPO il download con forza
            final_name = sanitize_filename(os.path.basename(original_path))
            final_path = os.path.join(output_dir, final_name)
            
            if os.path.exists(original_path):
                force_rename(original_path, final_path)
                print(f"SCARICATO: {final_name}")
            else:
                print(f"ERRORE: File non trovato - {original_path}")
    except Exception as e:
        print(f"ERRORE con {url}: {str(e)}")

def process_links_from_file(file_path='ai_learning/link.txt'):
    if not os.path.exists(file_path):
        print(f"ERRORE: File {file_path} non trovato")
        return
    
    with open(file_path, 'r') as file:
        links = [line.strip() for line in file if line.strip()]
    
    print(f"Trovati {len(links)} link da processare")
    for i, url in enumerate(links, 1):
        print(f"\nProcesso link {i}/{len(links)}")
        download_and_convert(url)
    
    # Pulizia finale per sicurezza
    audio_dir = os.path.join(os.path.dirname(file_path), 'audio')
    rename_all_wav_files(audio_dir)

if __name__ == "__main__":
    process_links_from_file()