import streamlit as st
import requests

# Configurar la página (debe ser lo primero que se ejecuta en Streamlit)
st.set_page_config(
    page_title="Recomendador de Cannabis",
    page_icon="🌿",
    layout="wide"
)

# URL de tu API desplegada
API_URL = "https://tu-api-fastapi.onrender.com"  # Cambia esto por tu URL real

def get_prediction(description):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"description": description}
        )
        return response.json()
    except Exception as e:
        return None

def main():
    # Título y descripción
    st.title("🌿 Recomendador de Cepas de Cannabis")
    st.markdown("### Encuentra la cepa perfecta para ti")

    # Área de entrada
    description = st.text_area(
        "Describe los efectos que estás buscando:",
        placeholder="Ejemplo: busco algo relajante y aromático para aliviar el estrés",
        height=100
    )

    # Botón de predicción
    if st.button("Obtener Recomendación", type="primary"):
        if description:
            # Obtener la recomendación desde la API de FastAPI
            result = obtener_recomendacion(description)

            if result:
                # Mostrar resultados
                st.success(f"✨ Cepa Recomendada: {result['strain_type']}")

                # Crear columnas para la información
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Nivel de Confianza", f"{result['confidence_percentage']}%")

                st.subheader("📝 Descripción")
                st.info(result['strain_description'])

                # Mostrar efectos y sabores
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("🎯 Efectos")
                    for effect in result['effects']:
                        st.write(f"• {effect}")

                with col4:
                    st.subheader("🌺 Sabores")
                    for flavor in result['flavors']:
                        st.write(f"• {flavor}")

                st.subheader("💡 Uso Recomendado")
                st.warning(result['recommended_use'])

        else:
            st.warning("⚠️ Por favor, ingresa una descripción")

    # Información adicional
    with st.expander("ℹ️ Información sobre tipos de cepas"):
        st.write("""
        - **Indica**: Conocida por sus efectos relajantes y calmantes.
        - **Sativa**: Proporciona efectos energizantes y estimulantes.
        - **Hybrid**: Combina características de ambas variedades.
        """)

if __name__ == "__main__":
    main()