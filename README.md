# 🌿 Plant Disease Detection using Evolutionary Algorithm

A web application that detects plant diseases from leaf images using **Genetic Algorithm** for feature selection and **Random Forest** for classification.

## 🎯 Features
- 🧬 Genetic Algorithm selects best 50 out of 112 features
- 🌲 Random Forest classifier with **94.23% accuracy**
- 📷 Upload image or use camera directly
- ⚖️ Compare two leaf images side by side
- 📄 Download PDF report of results
- 💊 Disease info, symptoms & treatment tips

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| DEAP | Genetic Algorithm |
| Scikit-learn | Random Forest classifier |
| OpenCV | Feature extraction |
| Streamlit | Web application |
| fpdf | PDF report generation |

## 📊 Model Performance
- **Training images:** 20,638
- **Classes:** 15 (Pepper, Potato, Tomato)
- **Features extracted:** 112
- **GA selected features:** 50
- **Final accuracy:** 94.23%

## 🧬 How It Works

```
Leaf Image
    → Extract 112 features (color, texture, shape)
    → Genetic Algorithm selects best 50 features
    → Random Forest predicts disease
    → Show result with treatment tips
```
## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/srilakshmi21206/plant-disease-ea.git
cd plant-disease-ea
```

### 2. Create virtual environment
```bash
python -m venv venv
call venv\Scripts\activate.bat
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```

### 5. Run the app
```bash
streamlit run app.py
```

## 🌱 Supported Plants & Diseases
| Plant | Diseases |
|-------|---------|
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy |

## 👩‍💻 Developed By
SriLLakshmi.k | JAIN University — Information Science & Engineering | 2026
