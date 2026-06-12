# disease_info.py

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial disease causing water-soaked spots on leaves and fruits.",
        "symptoms": "Small, dark brown spots with yellow halos on leaves.",
        "treatment": [
            "Apply copper-based bactericides",
            "Remove and destroy infected plant parts",
            "Avoid overhead irrigation",
            "Use disease-free seeds"
        ],
        "severity": "Medium",
        "emoji": "🫑"
    },
    "Pepper__bell___healthy": {
        "description": "Your pepper plant is healthy! No disease detected.",
        "symptoms": "No symptoms — plant looks great!",
        "treatment": [
            "Continue regular watering",
            "Maintain proper fertilization",
            "Monitor for early signs of disease"
        ],
        "severity": "None",
        "emoji": "✅"
    },
    "Potato___Early_blight": {
        "description": "Fungal disease caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings (target-board pattern).",
        "treatment": [
            "Apply fungicides like Chlorothalonil",
            "Remove infected leaves immediately",
            "Ensure proper plant spacing for air circulation",
            "Avoid wetting leaves when watering"
        ],
        "severity": "Medium",
        "emoji": "🥔"
    },
    "Potato___Late_blight": {
        "description": "Serious fungal disease caused by Phytophthora infestans.",
        "symptoms": "Water-soaked dark lesions on leaves, white mold on undersides.",
        "treatment": [
            "Apply Mancozeb or Metalaxyl fungicides",
            "Destroy infected plants completely",
            "Avoid planting in poorly drained soil",
            "Use resistant potato varieties"
        ],
        "severity": "High",
        "emoji": "🥔"
    },
    "Potato___healthy": {
        "description": "Your potato plant is healthy! No disease detected.",
        "symptoms": "No symptoms — plant looks great!",
        "treatment": [
            "Continue regular watering",
            "Maintain proper fertilization",
            "Monitor for early signs of disease"
        ],
        "severity": "None",
        "emoji": "✅"
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial disease caused by Xanthomonas species.",
        "symptoms": "Small, dark, water-soaked spots on leaves and fruits.",
        "treatment": [
            "Use copper-based sprays",
            "Remove infected plant debris",
            "Rotate crops annually",
            "Use certified disease-free seeds"
        ],
        "severity": "Medium",
        "emoji": "🍅"
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease caused by Alternaria solani.",
        "symptoms": "Brown spots with concentric rings, yellowing around spots.",
        "treatment": [
            "Apply Chlorothalonil fungicide",
            "Remove lower infected leaves",
            "Mulch around plants",
            "Water at base of plant only"
        ],
        "severity": "Medium",
        "emoji": "🍅"
    },
    "Tomato_Late_blight": {
        "description": "Devastating disease caused by Phytophthora infestans.",
        "symptoms": "Large, dark, greasy-looking spots on leaves and stems.",
        "treatment": [
            "Apply Mancozeb or Ridomil fungicide immediately",
            "Remove and bag infected plants",
            "Avoid working with wet plants",
            "Improve air circulation"
        ],
        "severity": "High",
        "emoji": "🍅"
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease caused by Passalora fulva.",
        "symptoms": "Pale yellow spots on upper leaf surface, olive-green mold below.",
        "treatment": [
            "Reduce humidity in greenhouse",
            "Apply fungicides like Copper Oxychloride",
            "Improve ventilation",
            "Remove infected leaves"
        ],
        "severity": "Medium",
        "emoji": "🍅"
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small circular spots with dark borders and light centers.",
        "treatment": [
            "Apply Mancozeb or Copper fungicides",
            "Remove infected lower leaves",
            "Avoid overhead watering",
            "Practice crop rotation"
        ],
        "severity": "Medium",
        "emoji": "🍅"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Pest infestation by Tetranychus urticae mites.",
        "symptoms": "Tiny yellow/white dots on leaves, fine webbing underneath.",
        "treatment": [
            "Apply Neem oil or Miticide spray",
            "Increase humidity around plants",
            "Introduce predatory mites",
            "Wash leaves with strong water spray"
        ],
        "severity": "Medium",
        "emoji": "🕷️"
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease caused by Corynespora cassiicola.",
        "symptoms": "Brown spots with concentric rings resembling a target.",
        "treatment": [
            "Apply Azoxystrobin fungicide",
            "Remove infected plant material",
            "Ensure good air circulation",
            "Avoid overhead irrigation"
        ],
        "severity": "Medium",
        "emoji": "🍅"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies.",
        "symptoms": "Yellowing and curling of leaves, stunted plant growth.",
        "treatment": [
            "Control whitefly population with insecticides",
            "Use reflective mulches to repel whiteflies",
            "Remove and destroy infected plants",
            "Use virus-resistant tomato varieties"
        ],
        "severity": "High",
        "emoji": "🍅"
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic patterns on leaves.",
        "symptoms": "Light and dark green mosaic pattern, leaf distortion.",
        "treatment": [
            "Remove and destroy infected plants",
            "Disinfect tools with bleach solution",
            "Wash hands before handling plants",
            "Use virus-resistant varieties"
        ],
        "severity": "High",
        "emoji": "🍅"
    },
    "Tomato_healthy": {
        "description": "Your tomato plant is healthy! No disease detected.",
        "symptoms": "No symptoms — plant looks great!",
        "treatment": [
            "Continue regular watering",
            "Maintain proper fertilization",
            "Monitor for early signs of disease"
        ],
        "severity": "None",
        "emoji": "✅"
    }
}

def get_disease_info(class_name):
    return DISEASE_INFO.get(class_name, {
        "description": "No information available.",
        "symptoms": "Unknown",
        "treatment": ["Consult a plant specialist"],
        "severity": "Unknown",
        "emoji": "🌿"
    })

def get_severity_color(severity):
    colors = {
        "None": "#2d6a4f",
        "Low": "#74c69d",
        "Medium": "#f4a261",
        "High": "#e63946",
        "Unknown": "#aaa"
    }
    return colors.get(severity, "#aaa")