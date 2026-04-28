# ✨ ALSKINCARE AI — Skincare Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black?style=for-the-badge&logo=flask)
![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> An end-to-end AI-powered skin analysis and personalized skincare recommendation system using deep learning and large language models.

---

## 🎬 Demo

![Demo](assets/demo.gif)

### Screenshots

| Hero Page | Analysis Results | AI Chat Assistant |
|-----------|-----------------|-------------------|
| ![Hero](assets/screenshot_hero.png) | ![Results](assets/screenshot_results.png) | ![Chat](assets/screenshot_chat.png) |

---

## 🚀 Features

- 🔬 **Multi-label skin condition detection** — 8 conditions simultaneously
- 🧠 **EfficientNet-B0 deep learning model** with ImageNet pretrained weights
- 📊 **Per-condition optimal threshold optimization** using ROC analysis
- 💬 **AI-powered chat assistant** using LLaMA 3.3 via Groq API
- 🌿 **Personalized morning/evening skincare routine** generation
- 📷 **Support for image upload and live camera capture**
- ⚖️ **Before & After skin comparison** feature
- 🎯 **Radar chart visualization** of all 8 conditions
- 🔒 **Privacy first** — images processed locally, never stored
- 📱 **Fully responsive modern dark UI**

---

## 🩺 Skin Conditions Detected

| Condition | Description | Common Causes |
|-----------|-------------|---------------|
| **Acne** | Inflammatory bumps, pimples, and pustules on the skin | Excess oil, clogged pores, bacteria, hormonal changes |
| **Blackheads** | Small, dark-colored bumps caused by clogged hair follicles | Excess sebum, dead skin cells, incomplete cleansing |
| **Dark Spots** | Hyperpigmented patches darker than surrounding skin | Sun exposure, post-inflammatory hyperpigmentation, aging |
| **Dryness** | Rough, flaky, or tight-feeling skin lacking moisture | Low humidity, harsh cleansers, dehydration, cold weather |
| **Normal** | Well-balanced skin with even tone and texture | Good hydration, balanced oil production, healthy barrier |
| **Oily** | Shiny, greasy skin with enlarged pores | Overactive sebaceous glands, genetics, humidity |
| **Pores** | Visibly enlarged pores, especially on nose and cheeks | Genetics, excess oil, aging, sun damage |
| **Wrinkles** | Fine lines and creases in the skin | Aging, UV exposure, dehydration, repetitive expressions |

---

## 📈 Model Performance

Final test set results after training:

| Condition | F1 Score | Precision | Recall | Accuracy |
|-----------|----------|-----------|--------|----------|
| Acne | 0.83 | 0.86 | 0.80 | 96% |
| Blackheads | 0.75 | 0.71 | 0.80 | 93% |
| Dark Spots | 0.84 | 0.81 | 0.87 | 96% |
| Dryness | 0.50 | 0.40 | 0.67 | 83% |
| Normal | 0.49 | 0.33 | 1.00 | 74% |
| Oily | 0.56 | 0.70 | 0.47 | 91% |
| Pores | 0.93 | 0.93 | 0.93 | 98% |
| Wrinkles | 0.87 | 0.87 | 0.87 | 97% |
| **Average** | **0.72** | **0.70** | **0.80** | **91%** |

> 📝 Model trained on CPU using EfficientNet-B0 with two-phase training strategy (freeze backbone → fine-tune).

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Deep Learning | PyTorch 2.1.0 | Model training and inference |
| Model Architecture | EfficientNet-B0 | Feature extraction with ImageNet weights |
| Backend | Flask 3.0.0 | REST API server |
| Frontend | HTML5, CSS3, JavaScript | Modern web interface |
| AI Chat | Groq + LLaMA 3.3 70B | Personalized recommendations |
| Image Processing | OpenCV + Pillow | Image preprocessing |
| Data Science | Scikit-learn, Pandas | Metrics and data handling |
| Visualization | Chart.js | Radar charts and graphs |

---

## 📁 Project Structure

```
ALSKINCARE-AI/
├── 📁 frontend/
│   └── index.html          # Complete SPA — HTML, CSS, JS
├── 📁 src/
│   ├── FixedSkinModel.py   # EfficientNet-B0 architecture
│   ├── pytorch_train.py    # Two-phase training pipeline
│   ├── pytorch_analyzer.py # Batch analysis tool
│   └── data_preprocessing.py # Dataset validation and splitting
├── 📁 models/
│   ├── fixed_skin_model.pth    # Trained model weights
│   └── optimal_thresholds.json # Per-condition thresholds
├── 📁 model_analysis/
│   ├── confusion_matrices.png  # Per-class confusion matrices
│   └── performance_metrics.png # F1, Precision, Recall charts
├── 📁 data/
│   └── skin_dataset.csv        # Dataset index file
├── app.py              # Flask backend + API endpoints
├── train.py            # Single command training entry point
├── config.py           # Centralized configuration
├── create_dataset.py   # Dataset CSV generator
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variables template
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/anshvishwakarmaaa/Skincare-Recommendation-System.git
cd Skincare-Recommendation-System
```

### 2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## 🔑 Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create a **free account**
3. Click **API Keys** → **Create API Key**
4. Copy the key to your `.env` file:
   ```
   GROQ_API_KEY=gsk_your_api_key_here
   ```
5. ✅ Free tier includes generous usage limits — more than enough for development and demos

---

## ▶️ Running the Application

### Option 1: Just run the app (model already trained)

```bash
python app.py
# Open browser at http://localhost:5000
```

### Option 2: Retrain the model from scratch

```bash
# Step 1: Generate dataset CSV from raw images
python create_dataset.py

# Step 2: Train the model
python train.py --epochs 20 --samples_per_class 100 --batch_size 8

# Step 3: Start the server
python app.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/analyze` | Analyze skin image (multipart/form-data) |
| `POST` | `/api/chat` | AI chat with skin context (JSON) |
| `GET` | `/api/conditions` | List all detectable conditions |

### Example: Analyze an image

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@my_photo.jpg"
```

### Example: Chat with AI

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What causes my acne?",
    "skin_context": {
      "skin_type": "Oily",
      "overall_score": 72,
      "detected_conditions": ["acne", "oily"]
    }
  }'
```

---

## 🧬 How It Works

1. 📸 **Upload** — User uploads skin photo via drag-drop or camera capture
2. 🔄 **Preprocess** — Image resized to 224×224, normalized with ImageNet stats
3. 🧠 **Extract** — EfficientNet-B0 extracts deep features from 4.8M parameters
4. 📊 **Classify** — Per-condition sigmoid outputs compared against optimal thresholds
5. 🎯 **Score** — Skin type determined, health score calculated (0–100)
6. 🤖 **Recommend** — Groq LLaMA generates personalized morning/evening routine
7. 💬 **Chat** — AI assistant answers follow-up questions using analysis context
8. 📱 **Display** — Results rendered with radar chart, condition cards, and routine steps

---

## 🏋️ Training Pipeline

The model uses a **two-phase training strategy** for optimal performance:

### Phase 1: Feature Extraction (Epochs 1–10)
- ❄️ Freeze entire EfficientNet-B0 backbone
- 🎯 Train only the custom classifier head
- 📈 Learning rate: `1e-3` with OneCycleLR scheduler

### Phase 2: Fine-Tuning (Epochs 11–20)
- 🔓 Unfreeze last 30 backbone layers
- 🎯 Fine-tune entire network end-to-end
- 📈 Learning rate: `1e-4` with CosineAnnealingWarmRestarts

### Training Details
- **Loss**: `BCEWithLogitsLoss` with class-weighted `pos_weight` for imbalance
- **Augmentation**: RandomResizedCrop, ColorJitter, GaussianBlur, RandomErasing
- **Early Stopping**: patience=10, monitoring validation F1
- **Threshold Optimization**: Sweep 0.1–0.9 per condition on validation set, select threshold maximizing per-class F1

---

## 📦 Dataset

- **Total Images**: ~11,500 across 8 skin conditions
- **Source**: Collected from dermatology image datasets
- **Preprocessing**: Validated, cleaned, and balanced to ~100 samples/class for training

### Per-Condition Distribution

| Condition | Image Count |
|-----------|-------------|
| Acne | 1,346 |
| Blackheads | 1,299 |
| Dark Spots | 1,345 |
| Dryness | 1,494 |
| Normal | 1,473 |
| Oily | 1,457 |
| Pores | 1,584 |
| Wrinkles | 1,504 |

> Images are multi-labeled — a single image may have multiple conditions (e.g., acne + oily).

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request

Please make sure to:
- Follow existing code style
- Add tests for new features
- Update documentation as needed

---

## 🔮 Future Improvements

- [ ] ☁️ Deploy to cloud (Render + Netlify)
- [ ] 🔍 Add Grad-CAM visual explanations
- [ ] 📱 Mobile app version (React Native)
- [ ] 🩺 Expand to 15+ skin conditions
- [ ] 🧴 Add product recommendation database
- [ ] 🌐 Multi-language support
- [ ] 👩‍⚕️ Dermatologist verification system
- [ ] 📊 Track skin progress over time
- [ ] 🧪 A/B testing for recommendation quality

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Ansh Vishwakarma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

- **[EfficientNet](https://arxiv.org/abs/1905.11946)** architecture by Google Brain
- **[LLaMA 3.3](https://ai.meta.com/llama/)** by Meta AI
- **[Groq](https://groq.com/)** for fast LLM inference
- **[PyTorch](https://pytorch.org/)** team for the deep learning framework
- **[torchvision](https://pytorch.org/vision/)** for pretrained model weights
- **[Chart.js](https://www.chartjs.org/)** for beautiful radar visualizations
- **[Google Fonts](https://fonts.google.com/)** — Inter & Plus Jakarta Sans

---

## 📬 Contact

- 👨‍💻 **Developer**: Ansh Vishwakarma
- 🐙 **GitHub**: [github.com/anshvishwakarmaaa](https://github.com/anshvishwakarmaaa)
- 📧 Feel free to open an [issue](https://github.com/anshvishwakarmaaa/Skincare-Recommendation-System/issues) for bugs or feature requests

---

<p align="center">
  Made with ❤️ and 🧠 by Ansh Vishwakarma
  <br>
  <sub>⭐ Star this repo if you found it helpful!</sub>
</p>