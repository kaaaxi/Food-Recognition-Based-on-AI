# FoodLens

**基于人工智能的食物识别与营养分析**

[English](README.md) | [简体中文](README_CN.md)

---

## 项目介绍

FoodLens 是一款基于人工智能的网页应用，能够从图片中识别食物并提供详细的营养分析。项目采用现代 MVC 架构，后端使用 FastAPI，前端使用 Vue 3，利用最先进的深度学习模型进行食物分类、图像分割和深度估计，以提供准确的营养信息。

**核心功能：**
- 上传或拍摄食物图片进行即时识别
- 多模型 AI 流水线确保准确的食物识别
- 详细的营养成分分析（卡路里、蛋白质、脂肪、碳水化合物）
- 历史记录追踪和饮食摄入统计
- 基于 JWT 的身份验证，支持匿名使用
- 响应式设计，移动端友好界面

## 项目结构

```
├── backend/
│   ├── app/                    # FastAPI MVC 应用
│   │   ├── views/              # 路由（API 端点）
│   │   ├── controllers/        # 请求处理与调度
│   │   ├── services/           # 业务逻辑与 AI 流水线
│   │   └── models/             # Pydantic 数据模型
│   └── models/                 # 预训练 AI 模型（需下载）
│       ├── Food101-Classifier/
│       ├── sam-hq-vit-base/
│       └── dpt-hybrid-midas/
├── src/                        # Vue 3 前端
└── public/                     # 静态资源
```

## AI 模型

本项目使用以下来自 Hugging Face 的预训练模型：

| 模型 | 用途 | Hugging Face 链接 |
|------|------|-------------------|
| **Food101-Classifier** | 食物菜名分类 | [VinnyVortex004/Food101-Classifier](https://huggingface.co/VinnyVortex004/Food101-Classifier) |
| **SAM-HQ** | 食物区域分割 | [syscv-community/sam-hq-vit-huge](https://huggingface.co/syscv-community/sam-hq-vit-huge/tree/main) |
| **DPT-Hybrid-MiDaS** | 深度估计（用于份量估算） | [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas/tree/main) |

**营养分析 LLM：** 阿里巴巴通义千问（通过 DashScope API）- 用于根据食物分类结果生成详细的营养 JSON 数据。

## 环境要求

### 系统要求
- Python 3.10+
- Node.js 18+
- 支持 CUDA 的 GPU（可选，用于加速推理）

### Python 依赖
```
fastapi>=0.100.0
uvicorn>=0.20.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=10.0.0
numpy>=1.24.0
openai>=1.0.0
python-multipart>=0.0.6
httpx>=0.24.0
safetensors>=0.4.0
timm>=0.9.0
python-dotenv>=1.0.0
```

### 环境变量配置

创建 `.env` 文件或设置以下环境变量：

| 变量名 | 是否必需 | 说明 |
|--------|----------|------|
| `SUPABASE_KEY` | 是 | Supabase 服务角色密钥（用于数据库） |
| `DASHSCOPE_API_KEY` | 是 | 阿里云 DashScope API 密钥（用于通义千问） |
| `SUPABASE_URL` | 否 | Supabase URL（已提供默认值） |
| `JWT_SECRET` | 否 | JWT 签名密钥（生产环境请修改） |


## 模型配置

下载预训练模型并放置到 `backend/models/` 目录：

## 运行应用

### 后端
```bash
uvicorn app.main:app --reload
```

### 前端
```bash
npm run dev
```

