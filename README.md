# CNN Modellerinin Pnömoni Tespitinde Karşılaştırılması

Göğüs röntgeni görüntülerinden zatürre tespiti yapan 6 farklı CNN modelini karşılaştırdığım dönem projem. `model_0` kendi tasarladığım model, `model_1` – `model_5` ise Kaggle'da bu görev için paylaşılmış başka mimariler.

## Veri Seti

Paul Mooney'in [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) veri setinin yeniden bölüştürülmüş bir versiyonunu kullandım.

| Set | Toplam | Normal | Lung Opacity |
|---|---|---|---|
| Train | 4,192 | 1,082 | 3,110 |
| Validation | 1,040 | 267 | 773 |
| Test | 624 | 234 | 390 |

Görüntüler 224×224 boyutuna küçültüldü ve grayscale (tek kanal) formatına çevrildi.

## Modeller

| Model | Açıklama |
|---|---|
| `model_0` | **Benim modelim** — 3 conv blok + büyük FC katman |
| `model_1` | 5 conv blok + Global Average Pooling |
| `model_2` | 4 conv blok + Global Average Pooling |
| `model_3` | BatchNorm/Dropout olmayan basit CNN |
| `model_4` | 3 katmanlı sade CNN |
| `model_5` | VGG benzeri derin CNN |

Her modelin kodu `models/` klasöründeki ayrı dosyalarda.

## Eğitim Ayarları

Adil karşılaştırma için tüm modeller aynı ayarlarla eğitildi:

- Optimizer: Adam (lr = 0.0001)
- Batch size: 32
- Epoch: 10
- Loss: CrossEntropyLoss (class weights: `[4.0, 0.5]` — sınıf dengesizliği için)
- Donanım: NVIDIA RTX 5070

## Sonuçlar

| Model | Test Accuracy | Normal Recall | Opacity Recall |
|---|---|---|---|
| model_0 | 83.65% | 0.64 | 0.96 |
| **model_1** | **86.22%** | **0.78** | 0.91 |
| model_2 | 82.69% | 0.67 | 0.92 |
| model_3 | 82.37% | 0.67 | 0.92 |
| model_4 | 84.29% | 0.62 | 0.98 |
| model_5 | 71.31% | 0.24 | 1.00 |

En iyi sonucu **model_1** verdi: hem test accuracy en yüksek, hem de sınıflar arası dengesi en iyi.

En kötü sonucu **model_5** verdi: en derin modelin olmasına rağmen overfit oldu, neredeyse her görüntüye opacity dedi.

Tüm eğitim grafikleri ve confusion matrix'ler `results/plots/` klasöründe.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook pneumonia_detected.ipynb
```

GPU kullanmak istiyorsanız PyTorch'u CUDA destekli kurun:
https://pytorch.org/get-started/locally/

## Proje Yapısı

```
pneumonia-cnn-comparison/
├── pneumonia_detected.ipynb    # Ana notebook
├── models/   # Her modelin .py dosyası
|   |
|   |__ README.md
|
├── results/
│   ├── logs/                   # Eğitim logları (.txt)
│   └── plots/                  # Grafikler ve confusion matrixler
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Kaynaklar

Karşılaştırdığım modellerin geldiği Kaggle notebook'ları:

- model_1: [https://www.kaggle.com/code/himel1122/final-project-computer-vison]
- model_2: [https://www.kaggle.com/code/akbarprdna/pneumonia-detection-from-chest-x-ray-using-cnn]
- model_3: [https://www.kaggle.com/code/rajasreerajamohanan/pneumonia-detection]
- model_4: [https://www.kaggle.com/code/abcdabcddsvgsgdsgf/nexpp]
- model_5: [https://www.kaggle.com/code/martinmauerer/pneumonia-xai]

## Yazar

[Muhammed Safa Özgökçe] — [Konya Teknik Üniversitesi / Bilgisayar Mühendisliği] dönem projesi
