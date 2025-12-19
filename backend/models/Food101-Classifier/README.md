---
license: mit
datasets:
- ethz/food101
metrics:
- accuracy
---
# Model Card for Model ID

This model is trained on the Food101 dataset and is designed to classify images of food into 101 different categories. It uses a ResNet-50 architecture and achieves high accuracy, making it suitable for image classification tasks in the food domain.

## Model Details

### Model Description

This model is a ResNet-50-based deep learning model, trained on the Food101 dataset. It classifies food images into 101 categories, such as Apple Pie, Cheesecake, Pizza, and others, using a convolutional neural network architecture that has been fine-tuned for optimal performance.


- **Developed by:** Vinny
- **Model type:** Image Classification (CNN - ResNet-50)
- **License:** MIT
- **Finetuned from model:** ResNet-50 pretrained on ImageNet

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

The model can be used for food image classification. Given an input image, it predicts one of 101 categories of food.


### Out-of-Scope Use

This model is trained specifically for food images and may not perform well on images outside of this domain. It should not be used for classifying non-food-related content. Malicious or unintended use such as misuse for surveillance should also be avoided.


## Bias, Risks, and Limitations

Since the model is trained on the Food101 dataset, its performance is limited to the classes within that dataset. If the food categories are skewed (e.g., over-representation of certain cuisines), this could affect performance on underrepresented categories.

### Risks

- **Class Imbalance:** Some food categories in the training data may have more examples than others, which could lead to biased predictions for overrepresented classes.
- **Misclassifications:** The model might misclassify food items that look similar to each other or if the image quality is poor.
- **Domain-specific Limitations:** This model is not generalized to recognize objects outside of food items.

        
### Recommendations

When using this model, users should ensure that the input images are of good quality and that the food items fall within the 101 categories of the Food101 dataset.

Further fine-tuning or retraining on more diverse datasets may be required if you want to improve classification for underrepresented classes.

## How to Get Started with the Model

Use the code below to get started with the model.


# Use the following command to install the required dependencies:

```bash
pip install torch torchvision
```

# Example Code

Below is an example of how to load and use the model for food image classification:

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = models.resnet50(pretrained=False, num_classes=101)
model.load_state_dict(torch.load('food101_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img = Image.open('your_image.jpg')  # replace with the path to your image
img_tensor = transform(img).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

# Map the predicted index to the food class label
food_classes = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
    'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
    'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings',
    'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast',
    'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
    'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta',
    'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli',
    'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
    'tuna_tartare', 'waffles'
]  # 101 food categories
predicted_class = food_classes[predicted.item()]

# Show the image and the prediction
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

```

## Training Details

### Training Data

The model was trained on the Food101 dataset, which contains 101,000 images across 101 food categories. Each category contains 1,000 images of various food types.

Dataset: Food101 Dataset

### Training Procedure

The model was trained using a ResNet-50 architecture. The following preprocessing steps were applied to the images:

#### Preprocessing [optional]

Resize images to 224x224 pixels.

Normalize pixel values to match the ImageNet statistics.

Convert images to tensors.


#### Training Hyperparameters

- **Epochs:** 10  
- **Batch size:** 32  
- **Learning rate:** 0.001  
- **Optimizer:** Adam

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics


#### Metrics

The model's performance was evaluated based on accuracy, with a final achieved accuracy of **95.03%** after 10 epochs.


### Results

Final Accuracy: 95.03%

The model shows excellent performance in classifying food images across 101 categories.


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).


**BibTeX:**

```
@misc{food101model,
  author = {VinnyVortex},
  title = {Food101 Image Classification Model},
  year = {2025},
  publisher = {Hugging Face}
}

```

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

ResNet-50: A convolutional neural network architecture that uses residual connections.

Food101: A dataset containing 101 food categories with 1,000 images each.
