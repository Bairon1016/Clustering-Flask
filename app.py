from flask import Flask, render_template, send_file
import Clustering
import os

app = Flask(__name__)


@app.route("/")
def inicio():
    data = Clustering.RealizarClustering(3)

    return render_template(
        "index.html",
        resultados=data["resultados"],
        resumen_cluster=data["resumen_cluster"],
        centroides=data["centroides"],
        mapa_sexo=data["mapa_sexo"],
        mapa_estado=data["mapa_estado"],
        mapa_recuperacion=data["mapa_recuperacion"]
    )


@app.route("/descargar-dataset")
def descargar_dataset():
    ruta = os.path.join(os.path.dirname(__file__), "Casos positivos de COVID-19 en Colombia.csv")
    return send_file(ruta, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
