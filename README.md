🌿 Cannabis Strain Recommendation System

A machine learning-powered web application that classifies cannabis strains into Sativa, Indica, or Hybrid categories based on user descriptions. Built with BERT transformer model and LoRA fine-tuning for optimal performance.

## Features

- **Classification**: Uses BERT-based transformer model for accurate strain prediction
- **Chat Interface**: Clean, user-friendly Gradio interface
- **Strain Recommendations**: Provides detailed strain information including effects, flavors, and descriptions
- **Real-time Processing**: Instant predictions with confidence scores
- **Multi-language Support**: Optimized for English descriptions

## Technologies

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Main programming language |
| **PyTorch** | Latest | Deep learning framework |
| **Transformers** | 4.x | Hugging Face transformers library |
| **Gradio** | Latest | Web interface framework |
| **Pandas** | Latest | Data manipulation and analysis |
| **PEFT** | Latest | Parameter Efficient Fine-Tuning |

### Machine Learning Stack

| Component | Details | Role |
|-----------|---------|------|
| **Base Model** | `bert-base-uncased` | Pre-trained transformer model |
| **Fine-tuning** | LoRA (Low-Rank Adaptation) | Efficient model adaptation |
| **Task Type** | Sequence Classification | Multi-class classification |
| **Classes** | Sativa, Indica, Hybrid | Cannabis strain categories |
| **Tokenizer** | BERT Tokenizer | Text preprocessing |

### Model Architecture

```
BERT Base Model (110M parameters)
├── Input: Text descriptions (max 256 tokens)
├── Transformer Layers: 12 layers
├── Hidden Size: 768
├── Attention Heads: 12
├── LoRA Adapters: Low-rank matrices
└── Output: 3 classes (Sativa, Indica, Hybrid)
```

## Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | Cannabis database/Kaggle |
| **Size** | 2,000+ strain records |
| **Features** | Strain name, type, effects, flavor, description |
| **Languages** | English |
| **Format** | CSV |

### Data Structure

| Column | Type | Description |
|--------|------|-------------|
| `Strain` | String | Name of the cannabis strain |
| `Type` | String | Category (sativa, indica, hybrid) |
| `Rating` | Float | User rating (1-5) |
| `Effects` | String | Concatenated effects (e.g., "CreativeEnergeticRelaxed") |
| `Flavor` | String | Concatenated flavors (e.g., "EarthySweetCitrus") |
| `Description` | String | Detailed strain description |
| `labels` | Integer | Encoded labels (0=Sativa, 1=Indica, 2=Hybrid) |

##  Model Performance


### Classification Results

| Metric | Sativa | Indica | Hybrid | Overall |
|--------|--------|--------|--------|---------|
| **Precision** | 0.85 | 0.82 | 0.78 | 0.82 |
| **Recall** | 0.83 | 0.79 | 0.81 | 0.81 |
| **F1-Score** | 0.84 | 0.80 | 0.79 | 0.81 |


### Features

 **Text Input**: Multi-line description field
 **Classification Button**: Instant strain classification
 **Results Display**: Formatted strain information with confidence scores
 **Example Prompts**: Pre-built descriptions for testing
 **Flavor Parsing**: Automatic separation of concatenated flavors
 **Effects Display**: Clean formatting of strain effects

## 🚀 Installation & Setup

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi
```

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/CannModel_en.git
cd CannModel_en
```

2. **Install Dependencies**
```bash
pip install torch transformers gradio pandas peft
```

3. **Prepare Data**
```bash
# Place your cannabis_clean.csv in the root directory
# Ensure it follows the required format
```

4. **Train Model** (Optional)
```bash
python src/train_model.py
```

5. **Run Application**
```bash
python src/gradio_app.py
```

##  Usage

### Web Interface

1. **Start Application**: Run `python src/gradio_app.py`
2. **Open Browser**: Navigate to `http://localhost:7860`
3. **Enter Description**: Type strain characteristics in the text field
4. **Get Prediction**: Click "Classify strain" button
5. **View Results**: See strain recommendation with details

### Example Descriptions

```text
"This strain provides relaxing effects and helps with sleep..."
"Energetic and uplifting strain that boosts creativity..."
"Balanced effects combining relaxation with mental clarity..."
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🌿 Built with ❤️ for the cannabis community**
