import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def ObtenerDatos():
    df = pd.read_csv("Casos positivos de COVID-19 en Colombia.csv")

    df = df[[
        "Edad", "Sexo", "Estado", "Recuperacion",
        "Inicio_Sintomas", "Fecha_Recuperacion"
    ]]

    # Convertir fechas correctamente
    df["Inicio_Sintomas"] = pd.to_datetime(df["Inicio_Sintomas"], dayfirst=True, errors='coerce')
    df["Fecha_Recuperacion"] = pd.to_datetime(df["Fecha_Recuperacion"], dayfirst=True, errors='coerce')

    # Crear días de recuperación
    df["Dias_Recuperacion"] = (
        df["Fecha_Recuperacion"] - df["Inicio_Sintomas"]
    ).dt.days

    # Eliminar datos inválidos
    df = df.dropna()
    df = df[df["Dias_Recuperacion"] >= 0]

    # Convertir categóricas
    df["Sexo"] = df["Sexo"].astype("category")
    df["Estado"] = df["Estado"].astype("category")
    df["Recuperacion"] = df["Recuperacion"].astype("category")

    mapa_sexo = dict(enumerate(df["Sexo"].cat.categories))
    mapa_estado = {0: "Fallecido", 1: "Leve"}
    mapa_recuperacion = {0: "Fallecido", 1: "Recuperado"}

    df["Sexo"] = df["Sexo"].cat.codes
    df["Estado"] = df["Estado"].cat.codes
    df["Recuperacion"] = df["Recuperacion"].cat.codes

    return df, mapa_sexo, mapa_estado, mapa_recuperacion


def RealizarClustering(nClusters=3):

    df, mapa_sexo, mapa_estado, mapa_recuperacion = ObtenerDatos()

    # 🔥 SOLO variables numéricas reales
    X = df[["Edad", "Dias_Recuperacion"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=nClusters, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X_scaled)

    df["Cluster"] = etiquetas

    # 🔥 IMPORTANTÍSIMO (centroides en escala real)
    centroides = scaler.inverse_transform(modelo.cluster_centers_).tolist()

    resultados = df.to_dict(orient="records")
    resumen_cluster = df["Cluster"].value_counts().to_dict()

    return {
        "resultados": resultados,
        "resumen_cluster": resumen_cluster,
        "centroides": centroides,
        "mapa_sexo": mapa_sexo,
        "mapa_estado": mapa_estado,
        "mapa_recuperacion": mapa_recuperacion
    }