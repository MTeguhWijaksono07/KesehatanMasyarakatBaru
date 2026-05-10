from groq import Groq
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY', '').strip("'\""))

def generate_answer(prompt, model='llama-3.3-70b-versatile', temperature=0.3, max_tokens=1000):
    """Fungsi sederhana untuk generate jawaban dari string prompt tunggal."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        return f"Maaf, terjadi kesalahan saat menghubungi AI: {str(e)}"

def refine_query(query, history):
    """
    Menghasilkan query standalone yang menggabungkan konteks percakapan sebelumnya.
    """
    if not history:
        return query
        
    context_msgs = []
    for msg in history[-4:]: # Cukup 2 pertukaran terakhir untuk menjaga fokus
        context_msgs.append(f"{msg['role'].upper()}: {msg['content']}")
    
    context_str = "\n".join(context_msgs)
    
    prompt = (
        "Tugas: Ubah PERTANYAAN TERBARU menjadi pertanyaan mandiri yang padat berdasarkan RIWAYAT.\n\n"
        "RIWAYAT:\n"
        f"{context_str}\n\n"
        f"PERTANYAAN TERBARU: {query}\n\n"
        "Aturan:\n"
        "1. JANGAN menjawab pertanyaan.\n"
        "2. Fokus pada subjek utama dari riwayat.\n"
        "3. Jika sudah mandiri, tulis ulang apa adanya.\n"
        "4. Hasil harus berupa satu kalimat pertanyaan.\n\n"
        "PERTANYAAN MANDIRI:"
    )
    
    standalone_query = generate_answer(prompt, temperature=0.0, max_tokens=100)
    return standalone_query.strip()

def call_llm(messages, model='llama-3.3-70b-versatile', temperature=0.3, max_tokens=1000, retries=3):
    """Panggil LLM dengan list messages dan retry otomatis."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f'Retry {attempt+1}/{retries} dalam {wait}s -- {e}')
                time.sleep(wait)
            else:
                print(f"Error in call_llm after {retries} attempts: {e}")
                return f"Maaf, terjadi kesalahan setelah beberapa kali mencoba: {str(e)}"

def check_llm_status():
    """Cek apakah LLM provider (Groq) aktif."""
    try:
        # Cek list model sebagai test koneksi ringan
        client.models.list()
        return True
    except Exception:
        return False
