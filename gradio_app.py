import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import json
import os
import pandas as pd
import random
from datetime import datetime

class CannabisClassifier:
    def __init__(self, model_path="results/final_model", csv_path="./cannabis_clean.csv"):
        self.model_path = model_path
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        self.cannabis_data = None
        
        # Cargar datos del CSV
        self.load_cannabis_data()
        # Cargar modelo y configuración
        self.load_model()
    
    def load_cannabis_data(self):
        """Cargar datos del CSV de cannabis"""
        try:
            self.cannabis_data = pd.read_csv(self.csv_path)
            print(f"✅ Datos de cannabis cargados: {len(self.cannabis_data)} cepas")
        except Exception as e:
            print(f"❌ Error cargando datos de cannabis: {e}")
            self.cannabis_data = None
    
    def load_model(self):
        """Cargar el modelo LoRA y tokenizer"""
        try:
            config_path = os.path.join(self.model_path, 'training_info.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.label_mapping = config.get('label_mapping', {'sativa': 0, 'indica': 1, 'hybrid': 2})
                model_name = config.get('model_name', 'bert-base-uncased')
            else:
                self.label_mapping = {'sativa': 0, 'indica': 1, 'hybrid': 2}
                model_name = 'bert-base-uncased'
            
            # Crear mapeo inverso
            self.id_to_label = {v: k for k, v in self.label_mapping.items()}

            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

            base_model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_mapping),
                output_attentions=False,
                output_hidden_states=False
            )
    
            # Cargar adaptadores LoRA
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f" Modelo cargado exitosamente desde {self.model_path}")
            print(f" Dispositivo: {self.device}")
            print(f" Clases: {list(self.label_mapping.keys())}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def get_strain_recommendation(self, predicted_type):
        """Obtener recomendación de cepa basada en el tipo predicho"""
        if self.cannabis_data is None:
            print("Cannabis data not loaded")
            return None
        
        # Mapear tipos a valores del CSV
        type_mapping = {'sativa': 'sativa', 'indica': 'indica', 'hybrid': 'hybrid'}
        csv_type = type_mapping.get(predicted_type.lower())
        
        print(f"Buscando cepas de tipo: {predicted_type} -> {csv_type}")
        
        if not csv_type:
            print(f"Tipo no reconocido: {predicted_type}")
            return None
        
        # Filtrar cepas por tipo
        matching_strains = self.cannabis_data[self.cannabis_data['Type'] == csv_type]
        print(f"Cepas encontradas: {len(matching_strains)}")
        
        if matching_strains.empty:
            print(f"No se encontraron cepas de tipo: {csv_type}")
            return None
        
        # Seleccionar una cepa aleatoria
        selected_strain = matching_strains.sample(n=1).iloc[0]
        print(f"Cepa seleccionada: {selected_strain['Strain']}")
        
        # Crear diccionario con todos los datos excepto rating
        strain_data = {
            'strain': selected_strain['Strain'],
            'type': selected_strain['Type'],
            'effects': selected_strain['Effects'],
            'flavor': selected_strain['Flavor'],
            'description': selected_strain['Description']
        }
        
        return strain_data
    def predict(self, description):
        """Predecir la cepa basada en la descripción"""
        if not description or description.strip() == "":
            return "Por favor ingresa una descripción válida.", None, None
        
        try:
            # Tokenizar
            inputs = self.tokenizer(
                description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Predicción
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Resultados
            predicted_label = self.id_to_label[predicted_class]
            confidence = probabilities[0][predicted_class].item()
            
            # Obtener recomendación de cepa
            strain_data = self.get_strain_recommendation(predicted_label)
            
            # Crear diccionario con todas las probabilidades

            all_probs = {}
            for label, idx in self.label_mapping.items():
                prob = probabilities[0][idx].item()
                all_probs[label.capitalize()] = float(f"{prob:.4f}")
            
            # Resultado principal con información de la cepa
            if strain_data:
                # Separar efectos y sabores con comas
                import re
                effects_separated = ', '.join(re.findall(r'[A-Z][a-z]+', strain_data['effects']))
                flavor_separated = ', '.join(re.findall(r'[A-Z][a-z]+', strain_data['flavor']))
                
                result = f"**Your strain: {strain_data['strain']}**\n\n"
                result += f"**Type: {strain_data['type'].upper()}**\n"
                result += f"**Confidence: {confidence:.1%}**\n\n"
                result += f"**Effects:** {effects_separated}\n\n"
                result += f"**Flavor:** {flavor_separated}\n\n"
                result += f"**Description:** {strain_data['description']}"
            else:
                result = f"**Prediction: {predicted_label.upper()}**\n"
                result += f"**Confidence: {confidence:.1%}**\n\n"
                result += "I can't find anything for you."
            
            return result, all_probs, strain_data
            
        except Exception as e:
            error_msg = f"Error en la predicción: {str(e)}"
            print(error_msg)
            return error_msg, None, None

# Inicializar clasificador
try:
    classifier = CannabisClassifier()
    model_loaded = True
except Exception as e:
    print(f"No se pudo cargar el modelo: {e}")
    model_loaded = False

def extract_top_effects(effects_string):
    """Extraer los primeros 3 efectos de la cadena de efectos"""
    if not effects_string or pd.isna(effects_string):
        return ["", "", ""]
    
    # Los efectos están concatenados, necesitamos separarlos
    # Buscar palabras que empiecen con mayúscula
    import re
    effects = re.findall(r'[A-Z][a-z]*', effects_string)
    
    # Asegurar que tengamos exactamente 3 efectos
    while len(effects) < 3:
        effects.append("")
    
    return effects[:3]

def predict_strain(description):
    """Funcio wrapper para Gradio"""
    if not model_loaded:
        return "Modelo no disponible. Asegurate de haber entrenado el modelo primero."
    
    result, probs, strain_data = classifier.predict(description)
    
    return result

# Ejemplos predefinidos
examples = [
    ["This strain provides a relaxing and calming effect, perfect for evening use. It helps with sleep and reduces anxiety."],
    ["Energetic and uplifting strain that boosts creativity and focus. Great for daytime activities and social situations."],
    ["Balanced effects combining relaxation with mental clarity. Good for both day and evening use, helps with mood."],
    ["Strong indica effects that provide deep body relaxation and pain relief. Best used before bedtime."],
    ["Pure sativa that delivers an energetic cerebral high with enhanced mood and creativity."]
]

# Crear interfaz Gradio
def create_interface():
    with gr.Blocks(
        title="🌿 Cannabis Strain recommendation",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Helvetica', sans-serif;
        }
        .output-text {
            font-size: 16px;
            line-height: 1.5;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🌿 Cannabis strain recomendattion system
        
        ### Please describe the strain you are looking for
                       
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                description_input = gr.Textbox(
                    label="BeRT Model",
                    placeholder="E.g: This strain provides relaxing effects and helps with sleep...",
                    lines=4,
                    max_lines=8
                )

                predict_btn = gr.Button(
                    "Classify strain",
                    variant="primary",
                    size="lg"
                )
                
                gr.Examples(
                    examples=examples,
                    inputs=description_input,
                    label="💡 Try this prompts"
                )
            
            with gr.Column(scale=2):
                result_output = gr.Markdown(
                    label="Prediction result",
                    elem_classes="output-text"
                )
        
                       
        # Conectar eventos
        predict_btn.click(
            fn=predict_strain,
            inputs=description_input,
            outputs=result_output
        )
        
        description_input.submit(
            fn=predict_strain,
            inputs=description_input,
            outputs=result_output
        )
    
    return interface

# Ejecutar la aplicación
if __name__ == "__main__":
    interface = create_interface()
    
    print("\n" + "="*50)
    print("🚀 Iniciando Cannabis Strain Classifier")
    print("="*50)
    
    if model_loaded:
        print("✅ Modelo cargado correctamente")
        print("🌐 Abriendo interfaz web...")
    else:
        print("⚠️  Modelo no encontrado - entrenar primero")
    
    interface.launch(
        server_name="localhost",  # Permite acceso desde otras IPs
        server_port=7860,
        show_api=True,
        share=False,  # Cambiar a True para link público
        inbrowser=True
    )