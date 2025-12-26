# FoodLens

**AI-Powered Food Recognition & Nutrition Analysis**

[English](README.md) | [简体中文](README_CN.md)

---

## Introduction

FoodLens is an AI-powered web application that recognizes food from images and provides detailed nutritional analysis. Built with a modern MVC architecture using FastAPI backend and Vue 3 frontend, it leverages state-of-the-art deep learning models for food classification, image segmentation, and depth estimation to deliver accurate nutrition information.

**Key Features:**
- Upload or capture food images for instant recognition
- Multi-model AI pipeline for accurate food identification
- Detailed nutritional breakdown (calories, protein, fat, carbs)
- History tracking and dietary intake statistics
- JWT-based authentication with optional anonymous usage
- Responsive design with mobile-friendly interface

## Project Structure

```
├── backend/
│   ├── app/                    # FastAPI MVC Application
│   │   ├── views/              # Routes (API endpoints)
│   │   ├── controllers/        # Request handling & dispatch
│   │   ├── services/           # Business logic & AI pipeline
│   │   └── models/             # Pydantic schemas
│   └── models/                 # Pre-trained AI models (download required)
│       ├── Food101-Classifier/
│       ├── sam-hq-vit-base/
│       └── dpt-hybrid-midas/
├── src/                        # Vue 3 Frontend
└── public/                     # Static assets
```

## AI Models

This project uses the following pre-trained models from Hugging Face:

| Model | Purpose | Hugging Face Link |
|-------|---------|-------------------|
| **Food101-Classifier** | Food dish classification | [VinnyVortex004/Food101-Classifier](https://huggingface.co/VinnyVortex004/Food101-Classifier) |
| **SAM-HQ** | Food region segmentation | [syscv-community/sam-hq-vit-huge](https://huggingface.co/syscv-community/sam-hq-vit-huge/tree/main) |
| **DPT-Hybrid-MiDaS** | Depth estimation for portion size | [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas/tree/main) |

**LLM for Nutrition Analysis:** Alibaba Qwen (via DashScope API) - Used to generate detailed nutritional JSON based on food classification results.

## Prerequisites

### System Requirements
- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (optional, for faster inference)

### Python Dependencies
```
fastapi==0.124.4
uvicorn==0.38.0
torch==2.2.2+cu121
torchvision==0.17.2+cu121
transformers==4.57.3
Pillow==11.3.0
numpy==1.26.4
openai==2.11.0
python-multipart==0.0.20
httpx==0.28.1
safetensors==0.7.0
timm==1.0.22
python-dotenv==1.2.1
```

### Environment Variables

Create a `.env` file or set the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_KEY` | Yes | Supabase service role key for database |
| `DASHSCOPE_API_KEY` | Yes | Alibaba DashScope API key for Qwen LLM |
| `SUPABASE_URL` | Optional | Supabase URL (default provided) |
| `JWT_SECRET` | Optional | JWT signing secret (change in production) |


## Model Setup

Download the pre-trained models and place them in `backend/models/`:

## Running the Application

### Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend
```bash
npm run dev
```
