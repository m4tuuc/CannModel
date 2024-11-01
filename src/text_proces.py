import re

def limpiar_texto(texto):
    """eliminando caracteres no deseados."""
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]', '', texto)
    return texto.lower().strip()
