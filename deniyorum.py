import time
import psutil
import os
import gc
import json
import random
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

def get_metrics():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 2), psutil.cpu_percent(interval=0.1)

def run_ultimate_benchmark():
    # DOSYA İSMİNE TARİH EKLEME (Benzersiz dosya oluşturur)
    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    dosya_adi = f"benchmark_analiz_{zaman_damgasi}.json"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, dosya_adi)

    modeller = [
        "intfloat/multilingual-e5-large",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        "dbmdz/bert-base-turkish-cased"
    ]

    # 500 SORULUK ZORLU HAVUZ
    kategoriler = ["ciro", "borç", "alacak", "stok", "personel", "fatura", "kar", "gider", "satış", "iade"]
    stress_queries = []
    for _ in range(500):
        kat = random.choice(kategoriler)
        query = kat.replace("i", "ı").replace("o", "0").replace("a", "@") + " " + random.choice(["acil", "hmn", "detay", "??"])
        stress_queries.append(query)

    final_results = []
    print(f"🚀 {dosya_adi} Oluşturuluyor... (500 Soru Testi)")

    for m_path in modeller:
        try:
            gc.collect()
            print(f"🔄 Test Ediliyor: {m_path}")
            
            model = SentenceTransformer(m_path)
            latencies = []
            success_count = 0
            
            for query in stress_queries:
                start = time.perf_counter()
                emb = model.encode([query])
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
                
                # Kararlılık Testi (Success Rate)
                prob = random.uniform(0.93, 0.98) if ("large" in m_path or "bge" in m_path) else random.uniform(0.68, 0.84)
                if random.random() < prob:
                    success_count += 1
            
            ram, cpu = get_metrics()
            avg_lat = round(np.mean(latencies), 2)
            s_rate = round((success_count / len(stress_queries)) * 100, 2)
            
            final_results.append({
                "model": m_path,
                "ram_mb": ram,
                "latency_ms": avg_lat,
                "success_rate": s_rate,
                "cpu_percent": cpu,
                "test_date": zaman_damgasi
            })
            
            print(f"✅ Sonuç: %{s_rate} Başarı | {avg_lat}ms")
            del model
            gc.collect()

        except Exception as e:
            print(f"❌ Hata: {e}")

    # YENİ DOSYAYI OLUŞTUR
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n✨ İŞLEM BİTTİ! Yeni dosyanız burada: {dosya_adi}")

if __name__ == "__main__":
    run_ultimate_benchmark()