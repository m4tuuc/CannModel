from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #depura posibles errores en GPU
print("CUDA disponible:", torch.cuda.is_available())



df = pd.read_csv('D:/PYTHON DATA/Recomendador/dataset/cannabis.csv')

# Filtrar las filas con clases necesarias ('sativa', 'indica', 'hibrida')
df = df[df['Type'].isin(['sativa', 'indica', 'hibrida'])]

# Mapear las etiquetas de clase a valores numéricos
label_mapping = {'sativa': 0, 'indica': 1, 'hibrida': 2}
df['labels'] = df['Type'].map(label_mapping)


df = df.dropna(subset=['Description'])

df['Description'] = df['Description'].astype(str)


df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])


# Asegurarse de que 'Rating' esté en formato int y dentro del rango de clases
df['Rating'] = df['Rating'].astype(int)

# Guardar el DataFrame limpio en un nuevo archivo CSV
clean_csv_path = 'D:/PYTHON DATA/Recomendador/dataset/cannabis_clean.csv'
df.to_csv(clean_csv_path, index=False)

#Cargar el dataset limpio con Hugging Face datasets
dataset = load_dataset('csv', data_files=clean_csv_path)

# Tokenizar el texto usando el tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

def preprocess_function(examples):
    return tokenizer(examples['Description'], padding="max_length", truncation=True)

# Tokenizar el dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Asegurarse de que 'labels' estén en long y las entradas correctas
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#  Dividir el dataset en entrenamiento y validación
train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2, seed=42)  # Añadir semilla para reproducibilidad
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

#  Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results/checkpoint',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",   # Guarda el modelo al final de cada época
    evaluation_strategy="epoch",
)


# Cargar el modelo con la capa de clasificación
model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=3)
model.to(device)
#inputs = {key: value.to(device) for key, value in inputs.items()}

# Usar el Trainer para entrenar y evaluar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador
save_path = "D:/PYTHON DATA/Recomendador/results/final_model"
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
