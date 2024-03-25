import os
from datetime import datetime


klasor_yolu = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part4'


cinsiyet_default = 0
ırk_default = 0


def yas_hesapla_ve_yeniden_adlandir(klasor_yolu, cinsiyet_default, ırk_default):
    for dosya in os.listdir(klasor_yolu):
        if dosya.endswith(".jpg"):
           
            parcalar = dosya.split('_')
            if len(parcalar) != 4:  # Beklenen format dışında isimlendirilmiş dosyaları atlar, yoksa hata veriyor
                print(f"{dosya} isim formatı beklenenin dışında.")
                continue

            try:
                dogum_tarihi = parcalar[2]
                cekim_yili = parcalar[3].split('.')[0]

                # Yaşı hesaplar
                dogum_yili, dogum_ayi, dogum_gunu = map(int, dogum_tarihi.split('-'))
                yas = int(cekim_yili) - dogum_yili
                if yas < 0:  # Yaş negatifse hata mesajı yazdırır
                    print(f"{dosya} için hesaplanan yaş negatif ({yas}).")
                    continue

               
                tarih_saat = datetime.now().strftime("%Y%m%d%H%M%S%f")
                yeni_ad = f"{yas}_{cinsiyet_default}_{ırk_default}_{tarih_saat}.jpg"
                
                
                eski_yol = os.path.join(klasor_yolu, dosya)
                yeni_yol = os.path.join(klasor_yolu, yeni_ad)
                os.rename(eski_yol, yeni_yol)
                print(f"{dosya} -> {yeni_ad}")

            except ValueError as e:
                print(f"{dosya} işlenirken bir hata oluştu: {e}")


yas_hesapla_ve_yeniden_adlandir(klasor_yolu, cinsiyet_default, ırk_default)