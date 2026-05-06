from flask import Flask, render_template
import Clustering

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


if __name__ == "__main__":
    app.run(debug=True)
