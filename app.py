from flask import Flask, render_template
import Clustering

app = Flask(__name__)

@app.route("/")
def inicio():
    info = Clustering.RealizarClustering()
    return render_template(
        "index.html",
        resultados=info["resultados"],
        resumen_cluster=info["resumen_cluster"],
        centroides=info["centroides"],
        mapa_sexo=info["mapa_sexo"],
        mapa_estado=info["mapa_estado"],
        mapa_recuperacion=info["mapa_recuperacion"],
    )

if __name__ == "__main__":
    app.run(debug=True)

# Mover todo a un template HTML para mostrar los resultados de forma más amigable
# Cargar un dataset de minería de datos desde Kaggle 
