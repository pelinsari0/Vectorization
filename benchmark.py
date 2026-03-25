import time, psutil, os, gc, json
import numpy as np
from sentence_transformers import SentenceTransformer

def get_metrics():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 2), psutil.cpu_percent(interval=0.1)

def run_ultra_stress_test():
    modeller = [
        "intfloat/multilingual-e5-large",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        "dbmdz/bert-base-turkish-cased"
    ]

    # 500 SORULUK DEV HAVUZ OLUŞTURMA (Kombinasyon Mantığı)
    kategoriler = ["ciro", "borç", "alacak", "stok", "personel", "satış", "fatura", "kar", "gider", "şube"]
    zamanlar = ["geçen yıl", "bu ay", "son çeyrek", "2025 yılı", "haftalık", "dünkü", "toplam", "güncel", "tahmini", "aylık"]
    tipler = ["nedir?", "durumu?", "raporu getir", "ayrıntısını göster", "listesi"]
    
    # 10 x 10 x 5 = 500 soru üretiliyor
    ultra_queries = [f"{z} {k} {t}" for k in kategoriler for z in zamanlar for t in tipler]

    final_results = []
    print(f"\n☢️ AŞAMA 4: 500 SORULUK ULTRA STRES TESTİ BAŞLATILDI ☢️\n")
    print(f"{'Model Adı':<42} | {'RAM (MB)':<10} | {'Sorgu (ms)':<10} | {'Toplam Sn':<10}")
    print("-" * 95)

    for m_path in modeller:
        try:
            gc.collect()
            model = SentenceTransformer(m_path)
            
            latencies = []
            success_count = 0
            total_start = time.perf_counter()
            
            for query in ultra_queries:
                start = time.perf_counter()
                emb = model.encode([query])
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)
                if emb is not None and not np.isnan(emb).any():
                    success_count += 1
            
            total_end = time.perf_counter()
            ram, cpu = get_metrics()
            avg_lat = round(np.mean(latencies), 2)
            total_time = round(total_end - total_start, 2)
            s_rate = (success_count / len(ultra_queries)) * 100
            
            res = {
                "model": m_path, "ram_mb": ram, "latency_ms": avg_lat,
                "total_time_sec": total_time, "cpu_percent": cpu, 
                "success_rate": s_rate, "query_count": 500
            }
            final_results.append(res)
            
            print(f"{m_path[:42]:<42} | {ram:<10} | {avg_lat:<10} | {total_time:<10}")
            del model
            gc.collect()

        except Exception as e:
            print(f"{m_path[:42]:<42} | HATA: {str(e)[:20]}")

    with open("benchmark_sonuclari.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 500 Soruluk Ultra Test Tamamlandı! Veriler 'benchmark_sonuclari.json' dosyasında.")

if __name__ == "__main__":
    run_ultra_stress_test()