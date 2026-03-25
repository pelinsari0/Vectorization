import json
import faiss
import numpy as np
import google.generativeai as genai
import time
import sys
import os

CHECKPOINT_FILE = "gemini_embeddings_checkpoint.json"
INPUT_FILE = "islenmis_veri.json"
OUTPUT_PREFIX = "index_"

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY tanımlı değil.")
        print("Önce terminalde API key ayarla.")
        sys.exit()
    return api_key

def configure_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)

def find_embedding_model():
    try:
        for m in genai.list_models():
            if "embedContent" in m.supported_generation_methods:
                return m.name
    except Exception as e:
        print(f"API bağlantı hatası: {e}")
        sys.exit()
    return None

def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_checkpoint(model_name, embeddings_list, last_completed_index, total_count):
    checkpoint_data = {
        "model_name": model_name,
        "last_completed_index": last_completed_index,
        "total_count": total_count,
        "embeddings": embeddings_list
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    print(f"Checkpoint kaydedildi -> son tamamlanan soru: {last_completed_index}")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(embeddings_list, model_name):
    print("\nTüm sorular tamamlandı, FAISS index oluşturuluyor...")
    embeddings_array = np.array(embeddings_list, dtype="float32")
    boyut = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(boyut)
    index.add(embeddings_array)

    dosya_adi_temiz = model_name.replace("/", "_")
    dosya_adi = f"{OUTPUT_PREFIX}{dosya_adi_temiz}.faiss"

    faiss.write_index(index, dosya_adi)
    print(f"BAŞARILI! FAISS index kaydedildi: {dosya_adi}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint dosyası silindi.")

def bulut_vektor_olustur_toplu():
    print("--- Bulut Modeli (Gemini API - Toplu İşlem) başlatıldı ---")

    configure_gemini()
    model_adi = find_embedding_model()

    if not model_adi:
        print("HATA: Uygun embedding modeli bulunamadı.")
        sys.exit()

    print(f"Kullanılan model: {model_adi}\n")

    data = load_data()
    sorular = [item["question"] for item in data]

    checkpoint = load_checkpoint()
    embeddings_list = []
    start_index = 0

    if checkpoint:
        if checkpoint.get("model_name") == model_adi and checkpoint.get("total_count") == len(sorular):
            embeddings_list = checkpoint.get("embeddings", [])
            start_index = checkpoint.get("last_completed_index", 0)
            print(f"Checkpoint bulundu. {start_index}. sorudan devam ediliyor.\n")
        else:
            print("Checkpoint mevcut ama veri/model uyuşmuyor. Sıfırdan başlanıyor.\n")

    paket_boyutu = 10
    print(f"Batch boyutu: {paket_boyutu}")
    print(f"Toplam soru: {len(sorular)}\n")

    for i in range(start_index, len(sorular), paket_boyutu):
        paket = sorular[i:i + paket_boyutu]

        try:
            sonuc = genai.embed_content(
                model=model_adi,
                content=paket,
                task_type="retrieval_document"
            )

            batch_embeddings = sonuc["embedding"]
            embeddings_list.extend(batch_embeddings)

            tamamlanan = i + len(paket)
            print(f"{tamamlanan} / {len(sorular)} soru tamamlandı")

            save_checkpoint(model_adi, embeddings_list, tamamlanan, len(sorular))

            time.sleep(5)

            if tamamlanan % 100 == 0 and tamamlanan != len(sorular):
                print("Rate limit koruması için 62 saniye bekleniyor...")
                time.sleep(62)

        except Exception as e:
            print(f"\nHATA: {i} - {i + len(paket)} arası sorularda sorun oluştu.")
            print(f"Ayrıntı: {e}")
            print("Muhtemelen kota doldu veya rate limit yedin.")
            print("Daha sonra yeni geçerli key ile tekrar çalıştırırsan kaldığı yerden devam eder.")
            sys.exit()

    build_faiss_index(embeddings_list, model_adi)

if __name__ == "__main__":
    bulut_vektor_olustur_toplu()