# 1_veri_hazirlayici.py
import json
import os

GIRDI_DOSYASI = 'golden_queries.json'
CIKTI_DOSYASI = 'islenmis_veri.json'

def veriyi_hazirla():
    print("Veri hazırlığı başlıyor...")
    
    if not os.path.exists(GIRDI_DOSYASI):
        print(f"Hata: {GIRDI_DOSYASI} bulunamadı!")
        return

    with open(GIRDI_DOSYASI, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Her veriye sabit bir ID atanarak vektörlerle referans bağı kurulur 
    for i, item in enumerate(data):
        item['vector_id'] = i

    with open(CIKTI_DOSYASI, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    soru_sayisi = len(data)
    print(f"Başarılı! Toplam {soru_sayisi} soru işlendi ve '{CIKTI_DOSYASI}' oluşturuldu.")

if __name__ == "__main__":
    veriyi_hazirla()