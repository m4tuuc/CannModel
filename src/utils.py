def mostrar_recomendaciones(df):
    """Muestra las recomendaciones de manera formateada."""
    for index, fila in df.iterrows():
        print(f"Cepa: {fila['Strain']}")
        print(f"Tipo: {fila['Type']}")
        print(f"Calificacion: {fila['Rating']}")
        print(f"Efectos: {fila['Effects']}")
        print(f"Sabor: {fila['Flavor']}")
        print(f"Descripción: {fila['Description']}")
        print("\n" + "-"*50 + "\n")
