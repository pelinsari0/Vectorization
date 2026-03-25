import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def eksik_modelleri_olustur():
    print("--- Eksik 4 Modelin Vektör Veritabanları Oluşturuluyor ---\n")
    
    # İşlenmiş veriyi okuyoruz
    with open('islenmis_veri.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    sorular = [item['question'] for item in data]

    # Menüde olup da henüz dosyası olmayan 4 model
    modeller = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        "dbmdz/bert-base-turkish-cased"
    ]

    for model_adi in modeller:
        print(f"\n---> [{model_adi}] indiriliyor ve vektörler hesaplanıyor (Bu biraz sürebilir)...")
        try:
            model = SentenceTransformer(model_adi)
            embeddings = model.encode(sorular, show_progress_bar=True)
            
            boyut = embeddings.shape[1]
            index = faiss.IndexFlatL2(boyut)
            index.add(np.array(embeddings).astype('float32'))
            
            dosya_adi = f"index_{model_adi.replace('/', '_')}.faiss"
            faiss.write_index(index, dosya_adi)
            print(f"BAŞARILI: {dosya_adi} dosyası oluşturuldu!")
        except Exception as e:
            print(f"HATA: {model_adi} modelinde sorun oluştu -> {e}")

    print("\n--- TÜM İŞLEMLER TAMAMLANDI! Artık Arayüzden Test Edebilirsiniz ---")

if __name__ == "__main__":
    eksik_modelleri_olustur()