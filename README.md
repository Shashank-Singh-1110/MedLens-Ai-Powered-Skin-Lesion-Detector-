# 🔬 MedLens — AI-Powered Skin Lesion Screening System

**Computer Vision + Explainable AI + RAG-Enhanced LLM for Dermatological Screening**

---

MedLens is an end-to-end AI screening tool that classifies dermoscopic skin lesion images into 7 diagnostic categories, generates Grad-CAM explainability heatmaps, and provides RAG-grounded medical explanations via a locally-running LLaMA 3 — all with zero cloud dependency.

> ⚠️ **Disclaimer:** MedLens is an educational/research project and is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist.

---

## ✨ Key Features

- **7-Class Skin Lesion Classification** — EfficientNet-B0 fine-tuned on HAM10000 (10,015 images)
- **Grad-CAM Heatmaps** — See exactly where the model focuses its attention
- **RAG-Enhanced Explanations** — LLaMA 3 generates responses grounded in verified medical knowledge
- **Confidence Margin Analysis** — Detects uncertain predictions (especially melanoma vs moles)
- **PDF Report Generation** — Downloadable report with images, heatmap, classification & explanation
- **Nearby Dermatologist Finder** — Google Maps integration for locating specialists
- **SSE Streaming** — Real-time word-by-word explanation delivery
- **100% Local Inference** — No patient data leaves the device

---

## 🏗️ Architecture

```
Image Upload
     │
     ▼
┌─────────────┐
│ Preprocess   │  Resize 224×224, Normalize (ImageNet stats)
└──────┬──────┘
       ▼
┌─────────────┐
│EfficientNet  │  7-class classification + confidence scores
│    B0        │
└──┬──────┬───┘
   │      │
   ▼      ▼
┌──────┐ ┌──────────┐
│Grad- │ │ Prompt   │  Confidence margin analysis
│ CAM  │ │ Adapter  │  + RAG knowledge retrieval
└──┬───┘ └────┬─────┘
   │          ▼
   │   ┌─────────────┐
   │   │  LLaMA 3    │  Streaming medical explanation
   │   │  (Ollama)   │  with safety rules
   │   └──────┬──────┘
   │          │
   ▼          ▼
┌─────────────────────┐
│   Frontend (SSE)    │  Real-time display + PDF report
│   + Google Maps     │  + Dermatologist finder
└─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| CV Model | EfficientNet-B0 (timm, PyTorch) |
| Augmentation | Albumentations |
| Explainability | Grad-CAM (custom implementation) |
| LLM | LLaMA 3 via Ollama |
| Knowledge Base | RAG with curated medical data |
| Backend | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS |
| PDF Reports | ReportLab |
| Maps | Google Maps Embed API |

---

## 📊 Dataset & Results

**Dataset:** [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 dermoscopic images, 7,470 unique lesions

| Class | Full Name | Severity | Count |
|-------|-----------|----------|-------|
| akiec | Actinic Keratoses | Pre-cancerous | 327 |
| bcc | Basal Cell Carcinoma | Cancerous | 514 |
| bkl | Benign Keratosis | Benign | 1,099 |
| df | Dermatofibroma | Benign | 115 |
| mel | Melanoma | Cancerous (dangerous) | 1,113 |
| nv | Melanocytic Nevi (Moles) | Benign | 6,705 |
| vasc | Vascular Lesions | Benign | 142 |

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **82.2%** |
| Val Accuracy | **84.7%** |
| Split Method | Patient-disjoint (by lesion_id) |
| Imbalance Ratio | 58.3x (nv:df) |

> We use **patient-disjoint splitting** — all images of the same lesion stay in the same split. This prevents data leakage and gives honest accuracy numbers. Papers reporting 90%+ typically use image-level splits.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with LLaMA 3 pulled
- Kaggle account (for dataset download)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/medlens.git
cd medlens
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Place your Kaggle API key at ~/.kaggle/kaggle.json
python download_dataset.py
```

### 3. Preprocess

```bash
python preprocess.py
```

This runs EDA, creates patient-disjoint train/val/test splits, resizes images to 224×224, and computes class weights.

### 4. Train

```bash
python train.py
```

Trains EfficientNet-B0 for 20 epochs with cosine annealing, oversampling, and label smoothing. Best model saved to `backend/models/best_model.pth`.

### 5. Start Ollama

```bash
ollama pull llama3
ollama serve
```

### 6. Start Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 7. Open Frontend

```bash
open frontend/index.html
```

---

## 📁 Project Structure

```
medlens/
├── config.py                  # Central configuration
├── download_dataset.py        # Kaggle dataset downloader
├── preprocess.py              # EDA + patient-disjoint splits
├── dataset.py                 # PyTorch dataset + augmentations + oversampling
├── train.py                   # Training loop with cosine annealing
├── requirements.txt
│
├── backend/
│   ├── app/
│   │   └── main.py            # FastAPI server (3 endpoints)
│   ├── models/
│   │   └── best_model.pth     # Trained weights (not in repo)
│   └── utils/
│       ├── inference.py       # Model loading + prediction
│       ├── gradcam.py         # Grad-CAM heatmap generation
│       ├── knowledge_base.py  # RAG medical knowledge store
│       ├── prompt_adapter.py  # CNN→LLM prompt construction
│       ├── llm.py             # Ollama streaming integration
│       └── report.py          # PDF report generator
│
├── frontend/
│   └── index.html             # Single-page clinical UI
│
└── data/
    ├── raw/                   # HAM10000 original (not in repo)
    └── processed/             # Train/val/test splits (not in repo)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status check |
| `POST` | `/analyze` | SSE stream — classification → heatmap → LLM explanation |
| `POST` | `/report` | Generate & download PDF report |
| `GET` | `/nearby-dermatologists` | Dermatologist search URL |

### Example: Analyze an image

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/skin_image.jpg"
```

Returns SSE stream:
```
data: {"type": "classification", "class_name": "Actinic Keratoses", "confidence": 0.914, ...}
data: {"type": "heatmap", "image": "<base64>"}
data: {"type": "llm_chunk", "content": "This"}
data: {"type": "llm_chunk", "content": " AI"}
...
data: {"type": "done"}
```

---

## 🔑 Key Technical Decisions

**Patient-Disjoint Splitting** — Split by `lesion_id` not `image_id`. One lesion can have multiple photos; naive splitting leaks data.

**Oversampling > Weighted Loss** — `WeightedRandomSampler` balances batches at the data level. Weighted loss destabilized training with extreme weights (66x).

**Confidence Margin Analysis** — When the gap between top-2 predictions is <15%, the system flags uncertainty. Critical for the melanoma vs moles confusion case.

**RAG over Fine-tuning** — Injecting verified medical facts into the prompt is faster, cheaper, and instantly updatable compared to fine-tuning the LLM.

**Local LLM** — Patient images never leave the device. Privacy by architecture, not by policy.

---

## 🔮 Future Scope

- [ ] Expand to ISIC 2019 dataset (33,569 images, 8 classes)
- [ ] Add U-Net lesion segmentation before classification
- [ ] Implement lesion history tracking over time
- [ ] Mobile deployment with React Native + ONNX runtime
- [ ] Multi-language support for rural healthcare access
- [ ] Fine-tune a medical-specific LLM
- [ ] ABCDE self-check interactive guide


---

## 📄 License

This project is for educational purposes. The HAM10000 dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## 🙏 Acknowledgments

- [ISIC Archive](https://www.isic-archive.com/) for the HAM10000 dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for EfficientNet implementation
- [Ollama](https://ollama.com) for local LLM inference
- [Albumentations](https://albumentations.ai/) for image augmentation
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [ReportLab](https://www.reportlab.com/) for PDF generation

---

<div align="center">

**Built with ❤️ for accessible healthcare**

*M.Tech Final Year Project — 2026*

</div>
