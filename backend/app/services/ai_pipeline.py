import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from openai import OpenAI
from PIL import Image

from ..config import settings


logger = logging.getLogger(__name__)


class AIPipeline:
    """End-to-end AI chain: Food101 Classification -> SAM-HQ Segmentation -> MiDaS Depth Estimation -> Qwen Nutrition JSON."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if AIPipeline._initialized:
            return
        AIPipeline._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AI Pipeline using device: {self.device}")

        self.food_labels = self._load_food101_labels()
        self.food_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        self.classifier = None
        self.sam_processor = None
        self.sam_model = None
        self.depth_processor = None
        self.depth_model = None
        
        try:
            self.classifier = self._load_classifier()
            logger.info("Food101 classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Food101 classifier: {e}")
        
        try:
            self.sam_processor, self.sam_model = self._load_sam()
            logger.info("SAM-HQ model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SAM-HQ model: {e}")
        
        try:
            self.depth_processor, self.depth_model = self._load_midas()
            logger.info("MiDaS depth model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load MiDaS model: {e}")

        # Read DashScope API key from environment variable
        import os
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            logger.warning("DASHSCOPE_API_KEY not set, LLM features will use fallback values")
            dashscope_key = "dummy-key-for-fallback"
        
        self.client = OpenAI(
            api_key=dashscope_key,
            base_url=settings.qwen_base_url,
        )

    def analyze_image(self, image_bytes: bytes, filename: str, user_id: str | None = None) -> Dict:
        dish_name, confidence = self._classify_food(image_bytes)
        mask_ratio, mask = self._segment_food(image_bytes)
        portion_grams = self._estimate_volume(image_bytes, mask_ratio, mask)
        nutrition = self._call_llm(dish_name, portion_grams)

        response = {
            "id": None,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "result": {
                "dish_name": dish_name,
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein", 0),
                "fat": nutrition.get("fat", 0),
                "carbs": nutrition.get("carbs", 0),
                "portion_grams": portion_grams,
                "confidence": confidence,
                "suggestions": nutrition.get("suggestions", []),
                "breakdown": nutrition,
                "alternatives": nutrition.get("alternatives", []),
                "meal_pairing": nutrition.get("meal_pairing", []),
            },
        }
        return response

    # --------------------- Classification ---------------------
    def _load_classifier(self) -> nn.Module:
        model_path = settings.models_root / "Food101-Classifier" / "food101_model.pth"
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.food_labels))
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        return model

    def _classify_food(self, image_bytes: bytes) -> Tuple[str, float]:
        if self.classifier is None:
            return "Unknown Food", 0.1
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.food_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = F.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)
            dish_name = self.food_labels[idx.item()] if idx is not None else "Unknown"
            confidence = round(conf.item(), 4)
            return dish_name.replace("_", " ").title(), confidence
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "Unknown Food", 0.1

    # --------------------- Segmentation ---------------------
    def _load_sam(self):
        from transformers import SamModel, SamProcessor
        model_dir = settings.models_root / "sam-hq-vit-base"
        processor = SamProcessor.from_pretrained(model_dir, use_safetensors=True)
        model = SamModel.from_pretrained(model_dir, use_safetensors=True).to(self.device)
        model.eval()
        return processor, model

    def _segment_food(self, image_bytes: bytes) -> Tuple[float, np.ndarray | None]:
        """Run SAM-HQ with a full-image box prompt; return coverage ratio and mask array."""
        if self.sam_processor is None or self.sam_model is None:
            return 0.38, None
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            box = [[0, 0, width, height]]
            inputs = self.sam_processor(image, input_boxes=[box], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
            )
            mask = masks[0][0].squeeze().cpu().numpy()
            coverage = float(mask.mean())
            return round(coverage, 4), mask
        except Exception as exc:
            logger.warning("SAM segmentation failed, fallback ratio 0.38: %s", exc)
            return 0.38, None

    # --------------------- Depth / Volume ---------------------
    def _load_midas(self):
        """Load MiDaS depth estimation model with multiple fallback strategies."""
        try:
            # Strategy 1: Try to use MiDaS from torch.hub (native PyTorch, no transformers security issue)
            logger.info("Loading MiDaS from torch.hub...")
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
            midas.eval().to(self.device)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.dpt_transform
            return transform, midas
        except Exception as e:
            logger.warning(f"torch.hub MiDaS failed: {e}")
        
        try:
            # Strategy 2: Try transformers with safetensors only
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info("Loading MiDaS from HuggingFace with safetensors...")
            processor = AutoImageProcessor.from_pretrained(
                "Intel/dpt-hybrid-midas",
                use_safetensors=True
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                "Intel/dpt-hybrid-midas",
                use_safetensors=True
            ).to(self.device)
            model.eval()
            return processor, model
        except Exception as e:
            logger.warning(f"HuggingFace MiDaS failed: {e}")
        
        logger.warning("All MiDaS loading strategies failed, depth estimation will use fallback")
        return None, None

    def _estimate_volume(self, image_bytes: bytes, mask_ratio: float, mask: np.ndarray | None) -> float:
        if self.depth_processor is None or self.depth_model is None:
            return round(max(60.0, 320 * mask_ratio), 2)
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(image)
            
            # Check if using torch.hub MiDaS (transform is callable) or transformers (processor has 'images' parameter)
            if callable(self.depth_processor) and not hasattr(self.depth_processor, 'from_pretrained'):
                # torch.hub MiDaS style
                input_batch = self.depth_processor(img_np).to(self.device)
                with torch.no_grad():
                    prediction = self.depth_model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_np.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_np = prediction.cpu().numpy()
            else:
                # transformers style
                inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.depth_model(**inputs)
                    predicted_depth = outputs.predicted_depth
                depth = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_np = depth.squeeze().cpu().numpy()
            
            depth_np = depth_np - depth_np.min()
            if depth_np.max() > 0:
                depth_np = depth_np / depth_np.max()
            if mask is not None and mask.shape == depth_np.shape:
                region_depth = depth_np * mask
                mean_depth = region_depth.sum() / (mask.sum() + 1e-6)
            else:
                mean_depth = float(depth_np.mean())
            base_weight = 280.0 * mask_ratio
            portion_grams = max(40.0, base_weight * (0.8 + 0.6 * mean_depth))
            return round(portion_grams, 2)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Depth estimation failed, fallback to heuristic: %s", exc)
            return round(max(60.0, 320 * mask_ratio), 2)

    # --------------------- LLM ---------------------
    def _call_llm(self, dish_name: str, portion_grams: float) -> Dict:
        system_prompt = """You are an expert nutrition analyst. Given a dish name and estimated portion in grams,
provide accurate nutritional information and personalized dietary suggestions.

IMPORTANT: You must return a valid JSON object with the following structure:
{
    "calories": <number>,
    "protein": <number in grams>,
    "fat": <number in grams>,
    "carbs": <number in grams>,
    "fiber": <number in grams>,
    "sodium": <number in mg>,
    "suggestions": [<3 personalized dietary suggestions as strings>],
    "alternatives": [<3 healthier food alternatives as strings>],
    "meal_pairing": [<2-3 complementary dishes for balanced nutrition>]
}

Base your estimates on standard nutritional databases. Consider the portion size when calculating values.
Suggestions should be specific and actionable for improving dietary balance."""

        user_prompt = f"""Analyze this food item:
- Dish Name: {dish_name}
- Estimated Portion: {portion_grams} grams

Provide complete nutritional breakdown and personalized recommendations."""

        try:
            completion = self.client.chat.completions.create(
                model=settings.qwen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=12,
            )
            content = completion.choices[0].message.content
            return self._safe_parse_json(content)
        except Exception as exc:
            logger.warning("LLM call failed, fallback heuristic: %s", exc)
            calories = round(portion_grams * 1.6, 1)
            protein = round(portion_grams * 0.08, 1)
            fat = round(portion_grams * 0.05, 1)
            carbs = round(portion_grams * 0.12, 1)
            return {
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "suggestions": [
                    "Add more vegetables for fiber boost.",
                    "Control oil and salt, choose light cooking methods.",
                    "Pair with sugar-free beverages to avoid extra calories.",
                ],
            }

    def _safe_parse_json(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Non-JSON LLM response received, text=%s", text)
            return {}

    # --------------------- Helpers ---------------------
    def _load_food101_labels(self) -> list[str]:
        # Food-101 official class list
        return [
            "apple_pie",
            "baby_back_ribs",
            "baklava",
            "beef_carpaccio",
            "beef_tartare",
            "beet_salad",
            "beignets",
            "bibimbap",
            "bread_pudding",
            "breakfast_burrito",
            "bruschetta",
            "caesar_salad",
            "cannoli",
            "caprese_salad",
            "carrot_cake",
            "ceviche",
            "cheesecake",
            "cheese_plate",
            "chicken_curry",
            "chicken_quesadilla",
            "chicken_wings",
            "chocolate_cake",
            "chocolate_mousse",
            "churros",
            "clam_chowder",
            "club_sandwich",
            "crab_cakes",
            "creme_brulee",
            "croque_madame",
            "cup_cakes",
            "deviled_eggs",
            "donuts",
            "dumplings",
            "edamame",
            "eggs_benedict",
            "escargots",
            "falafel",
            "filet_mignon",
            "fish_and_chips",
            "foie_gras",
            "french_fries",
            "french_onion_soup",
            "french_toast",
            "fried_calamari",
            "fried_rice",
            "frozen_yogurt",
            "garlic_bread",
            "gnocchi",
            "greek_salad",
            "grilled_cheese_sandwich",
            "grilled_salmon",
            "guacamole",
            "gyoza",
            "hamburger",
            "hot_and_sour_soup",
            "hot_dog",
            "huevos_rancheros",
            "hummus",
            "ice_cream",
            "lasagna",
            "lobster_bisque",
            "lobster_roll_sandwich",
            "macaroni_and_cheese",
            "macarons",
            "miso_soup",
            "mussels",
            "nachos",
            "omelette",
            "onion_rings",
            "oysters",
            "pad_thai",
            "paella",
            "pancakes",
            "panna_cotta",
            "peking_duck",
            "pho",
            "pizza",
            "pork_chop",
            "poutine",
            "prime_rib",
            "pulled_pork_sandwich",
            "ramen",
            "ravioli",
            "red_velvet_cake",
            "risotto",
            "samosa",
            "sashimi",
            "scallops",
            "seaweed_salad",
            "shrimp_and_grits",
            "spaghetti_bolognese",
            "spaghetti_carbonara",
            "spring_rolls",
            "steak",
            "strawberry_shortcake",
            "sushi",
            "tacos",
            "takoyaki",
            "tiramisu",
            "tuna_tartare",
            "waffles",
        ]
