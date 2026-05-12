# Modeller

Bu klasörde karşılaştırdığım 6 CNN mimarisinin kodu var. Her dosya bir model tanımı içeriyor.

- `model_0.py` — Benim tasarladığım model
- `model_1.py` ile `model_5.py` — Kaggle'da paylaşılmış başka mimariler

Hepsinin sınıf adı `PneumoniaCNN`. Notebook'tan kullanmak için:

```python
from models.model_1 import PneumoniaCNN
model = PneumoniaCNN().to(device)
```

Tüm modeller 1 kanallı (grayscale), 224×224 girdi alır ve 2 sınıf logit çıkarır.
