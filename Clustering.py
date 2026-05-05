import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def ObtenerDatos():
    df = pd.read_csv("Casos positivos de COVID-19 en Colombia.csv")

    df = df[["Edad", "Sexo", "Estado", "Recuperacion"]]

    df = df.dropna()

    # Convertir a categorías
    df["Sexo"] = df["Sexo"].astype("category")
    df["Estado"] = df["Estado"].astype("category")
    df["Recuperacion"] = df["Recuperacion"].astype("category")

    # Mapas (para mostrar en HTML)
    mapa_sexo = dict(enumerate(df["Sexo"].cat.categories))
    mapa_estado = dict(enumerate(df["Estado"].cat.categories))
    mapa_recuperacion = dict(enumerate(df["Recuperacion"].cat.categories))

    # Convertir a números
    df["Sexo"] = df["Sexo"].cat.codes
    df["Estado"] = df["Estado"].cat.codes
    df["Recuperacion"] = df["Recuperacion"].cat.codes

    return df, mapa_sexo, mapa_estado, mapa_recuperacion


def RealizarClustering(nClusters=3):

    df, mapa_sexo, mapa_estado, mapa_recuperacion = ObtenerDatos()

    X = df[["Edad", "Sexo", "Estado", "Recuperacion"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=nClusters, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X_scaled)

    df["Cluster"] = etiquetas

    resultados = df.to_dict(orient="records")
    resumen_cluster = df["Cluster"].value_counts().to_dict()
    centroides = modelo.cluster_centers_.tolist()

    return {
        "resultados": resultados,
        "resumen_cluster": resumen_cluster,
        "centroides": centroides,
        "mapa_sexo": mapa_sexo,
        "mapa_estado": mapa_estado,
        "mapa_recuperacion": mapa_recuperacion
    }