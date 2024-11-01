# Recomendador de Cepas de Cannabis basados en imput (TEST)

Este proyecto es una API basado en FastAPI que recomienda cepas de cannabis basadas en descripciones proporcionadas por el usuario. Utiliza un modelo BERT entrenado para clasificar y predecir la cepa más relevante según la descripción ingresada.

La eleccion de este modelo se basa en buscar una mejor entendimiento del contexto de lo que el usuario ingresa.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Endpoints](#endpoints)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Tecnologias Utilizadas](#tecnologías-utilizadas)
- [Licencia](#licencia)

## Descripción

Esta API utiliza procesamiento de lenguaje natural (NLP) para identificar cepas de cannabis que correspondan con la descripción de sus efectos, aromas o beneficios médicos. El modelo BERT clasifica cada descripción en una de las cepas aprendidas durante el entrenamiento.

## Características

- Recomendación de cepas basada en descripciones detalladas.
- Documentación interactiva de API en `/docs`.
- Implementación con soporte para despliegue en plataformas en la nube.

## Requisitos Previos

Antes de comenzar, asegúrate de tener los siguientes programas instalados:
- [Requeriments.txt]
- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

Es recomendable crear un entorno virtual para manejar las dependencias de forma aislada.

## Instalación

1. **Clona este repositorio**:

   ```bash
   git clone https://github.com/tu-usuario/recomendador-cannabis.git
   cd recomendador-cannabis

## Instala las dependencias:
pip install -r requirements.txt

Si usas un entorno virtual:
python -m venv venv
source venv/bin/activate


## Uso
Ejecuta la API:

`python uvicorn main:app --reload`

La API estara disponible en http://127.0.0.1:8000, la documentación interactiva se encuentra en http://127.0.0.1:8000/docs.


## Estructura del Proyecto

├── dataset # Dataset de entrenamiento │ └── cannabis_clean.csv # CSV con datos procesados ├── src # Código fuente de la aplicación │ ├── main.py # Archivo principal de la API │ └── model.py # Código para cargar y usar el modelo ├── results # Directorio para almacenar el modelo entrenado │ └── checkpoint-171 # Carpeta con el modelo BERT entrenado ├── requirements.txt # Dependencias del proyecto └── README.md # Documentación del proyecto




## Endpoints
*POST* /predict
Este endpoint recibe una descripcion en formato JSON y devuelve la cepa recomendada que más se ajusta a esa descripción.

URL: /predict

Método: POST

Entrada:

JSON con la descripción de la cepa.


## Ejemplos de Uso
Puedes hacer una solicitud a la API usando curl o una herramienta como Postman.

Ejemplo con curl
 `python
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"description\": \"Can assist with inflammation, irritability, and minor physical discomfort.\"}" `

Ejemplo con requests en Python
 `python
import requests
url = "http://127.0.0.1:8000/predict"
payload = {
    "description": "Can assist with inflammation, irritability, and minor physical discomfort." 
}
response = requests.post(url, json=payload) `

 `python print(response.json())`





## Tecnologias Utilizadas
Python |
FastAPI |
Uvicorn |
Transformers (Hugging Face): BERT. |
Scikit-Learn |
Torch
