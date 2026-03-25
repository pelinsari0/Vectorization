# 2_lokal_vektor_olusturucu.py
import json
import faiss
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

def lokal_vektor_olustur(model_adi):
    print(f"--- {model_adi} için işlem başlatıldı ---")
    
    # İşlenmiş ve ID atanmış veriyi oku 
    with open('islenmis_veri.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sorular = [item['question'] for item in data]
    
    # Modeli Yükle
    print("Model HuggingFace üzerinden indiriliyor/yükleniyor...")
    model = SentenceTransformer(model_adi)
    
    # Vektörleştirme
    print("Sorular vektörleştiriliyor...")
    embeddings = model.encode(sorular, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # FAISS veritabanına kaydetme 
    boyut = embeddings.shape[1]
    index = faiss.IndexFlatL2(boyut)
    index.add(embeddings)
    
    # Dosya adındaki '/' gibi karakterleri temizle
    dosya_adi = f"index_{model_adi.replace('/', '_')}.faiss"
    faiss.write_index(index, dosya_adi)
    
    print(f"Başarılı! Vektör havuzu '{dosya_adi}' adıyla kaydedildi.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lokal Embedding Model Çalıştırıcı")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model adı")
    args = parser.parse_args()
    
    lokal_vektor_olustur(args.model)