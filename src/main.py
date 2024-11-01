from transformers import BertForSequenceClassification, BertTokenizer
from data_loader import load_data
from text_proces import limpiar_texto
from bert_model import BERTEmbedding
from recomendador import RecomendadorCannabis
from utils import mostrar_recomendaciones


def main():
    #Cargar y limpiar los datos
    ruta_datos = 'D:\PYTHON DATA\Recomendador\dataset\cannabis_clean.csv'
    df = load_data(ruta_datos)

    # Asegurarnos de que todos los valores de la columna 'Description' sean cadenas
    df['Description'] = df['Description'].fillna('').astype(str)

    # Limpiar las descripciones
    df['Description'] = df['Description'].apply(limpiar_texto)

    # Inicializar modelo
    #modelo_bert = BERTEmbedding()
    model_path = "D:\PYTHON DATA\Recomendador\src\results\checkpoint-171"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')


    # Inicializar recomendador
    recomendador = RecomendadorCannabis(df, model)

    # Input del usuario
    entrada_usuario = input("Describe qué tipo de cannabis te gustaría: ")
    entrada_usuario = limpiar_texto(entrada_usuario)

    # Obtener recomendaciones
    recomendaciones = recomendador.recomendar(entrada_usuario)

    # Mostrar recomendaciones
    mostrar_recomendaciones(recomendaciones)


if __name__ == "__main__":
    main()


