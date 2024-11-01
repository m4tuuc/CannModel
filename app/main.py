import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Modificar la carga del modelo para usar rutas relativas o almacenamiento en la nube

# Usar un modelo pre-entrenado de BERT
print("Cargando modelo...")
model = AutoModelForSequenceClassification.from_pretrained('D:\PYTHON DATA\Recomendador\src\results\checkpoint-171')
tokenizer = AutoTokenizer.from_pretrained('D:\PYTHON DATA\Recomendador\src\results\checkpoint-171')
print("Modelo cargado exitosamente")

class StrainInput(BaseModel):
    description: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "relajante y aromatica"
            }
        }

class StrainPrediction(BaseModel):
    strain_type: str
    confidence_percentage: float
    strain_description: str
    effects: list[str]
    flavors: list[str]
    recommended_use: str

@app.post("/predict", response_model=StrainPrediction)
async def predict_strain(input_data: StrainInput):
    try:
        # Tokenizar la entrada
        inputs = tokenizer(
            input_data.description,
            return_tensors='pt',
            truncation=True,
            padding=True
        )

        # Realizar la clasificación
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Calcular la predicción y confianza
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence_percentage = float(probabilities[predicted_class_idx] * 100)

        # Información detallada de las cepas
        strain_info = {
            0: {
                "type": "Indica",
                "description": "Las cepas Indica son conocidas por sus efectos relajantes y calmantes. Son ideales para el uso nocturno y para aliviar el estrés.",
                "effects": ["Relajante", "Sedante", "Reduce el dolor", "Induce el sueño"],
                "flavors": ["Terroso", "Dulce", "Pino"],
                "recommended_use": "Mejor para uso nocturno y para aliviar el dolor crónico"
            },
            1: {
                "type": "Sativa",
                "description": "Las Sativas son energizantes y estimulantes mentalmente. Perfectas para uso diurno y actividades creativas.",
                "effects": ["Energético", "Eufórico", "Creativo", "Estimulante"],
                "flavors": ["Cítrico", "Tropical", "Afrutado"],
                "recommended_use": "Ideal para uso diurno y actividades sociales"
            },
            2: {
                "type": "Hybrid",
                "description": "Los híbridos combinan efectos de Indica y Sativa, ofreciendo un equilibrio entre relajación y energía.",
                "effects": ["Equilibrado", "Relajante suave", "Estimulante moderado"],
                "flavors": ["Variado", "Frutal", "Especiado"],
                "recommended_use": "Versátil, adecuado tanto para uso diurno como nocturno"
            }
        }

        strain_data = strain_info.get(predicted_class_idx, {
            "type": "Unknown",
            "description": "Información no disponible",
            "effects": [],
            "flavors": [],
            "recommended_use": "No disponible"
        })

        return StrainPrediction(
            strain_type=strain_data["type"],
            confidence_percentage=round(confidence_percentage, 2),
            strain_description=strain_data["description"],
            effects=strain_data["effects"],
            flavors=strain_data["flavors"],
            recommended_use=strain_data["recommended_use"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Recomendación de Cannabis3"}