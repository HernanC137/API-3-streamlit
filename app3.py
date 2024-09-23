# Importamos las bibliotecas necesarias
import uvicorn
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
import tempfile  # Biblioteca para crear archivos temporales
import shutil  # Biblioteca para copiar archivos
from pycaret.classification import predict_model

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "ridge_model.pkl"
with open(model_path, 'rb') as model_file:
    dt2 = pickle.load(model_file)

# Cargar base de predicción en kaggle (si lo requieres, sino omite la carga aquí)
prueba = pd.read_csv("prueba_APP.csv")

# Definir un endpoint para manejar la subida de archivos y hacer predicciones
@app.post("/upload-excel")
def upload_excel(file: UploadFile = File(...)):
    try:
        # Crear un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            # Leer el archivo Excel usando pandas y almacenarlo en un DataFrame
            df = pd.read_excel(temp_file.name)

            # Preparar las variables eliminando las columnas no necesarias
            base_modelo2 = df.drop(columns=['Email', 'price'])

            # Realizar predicción
            df_test = base_modelo2.copy()
            predictions = predict_model(dt2, data=df_test)
            predictions["Precio"] = predictions["prediction_label"]
            prediction_label = list(predictions["Precio"])

            return {"predictions": prediction_label}

    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}

# Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

