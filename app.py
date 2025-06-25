from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
import os
import warnings

# Suprimir warnings de sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Rutas de los archivos
MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Inicializar variables globales
model = None
scaler = None

# Función para cargar modelo y escalador
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modelo o scaler no encontrados.")
    
    model_loaded = joblib.load(MODEL_PATH)
    scaler_loaded = joblib.load(SCALER_PATH)
    
    return model_loaded, scaler_loaded

# Intentar cargar modelo y escalador al iniciar
try:
    model, scaler = load_model_and_scaler()
    logging.info("✅ Modelo y scaler cargados correctamente.")
    
    # Verificar qué características espera el scaler
    if hasattr(scaler, 'feature_names_in_'):
        logging.info(f"🏷️ Características esperadas por el scaler: {list(scaler.feature_names_in_)}")
    if hasattr(scaler, 'n_features_in_'):
        logging.info(f"🔢 Número de características esperadas: {scaler.n_features_in_}")
    if hasattr(scaler, 'mean_'):
        logging.info(f"📊 Media del scaler: {scaler.mean_}")
    if hasattr(scaler, 'scale_'):
        logging.info(f"📏 Escala del scaler: {scaler.scale_}")
        
except Exception as e:
    logging.error(f"❌ Error al cargar el modelo o el escalador: {str(e)}")

@app.route('/')
def home():
    logging.info("🏠 Acceso a página principal")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Logging de la petición recibida
        logging.info("🔍 Petición POST recibida en /predict")
        logging.info(f"📝 Datos del formulario: {dict(request.form)}")
        
        # Obtener datos del formulario
        monthly_hours = float(request.form['MonthlyHours'])
        fan = float(request.form['Fan'])
        television = float(request.form['Television'])
        tariff_rate = float(request.form['TariffRate'])
        monitor = float(request.form['Monitor'])
        refrigerator = float(request.form['Refrigerator'])

        logging.info(f"📊 Datos procesados:")
        logging.info(f"   - Horas mensuales: {monthly_hours}")
        logging.info(f"   - Ventilador: {fan}")
        logging.info(f"   - Televisión: {television}")
        logging.info(f"   - Tarifa: {tariff_rate}")
        logging.info(f"   - Monitor: {monitor}")
        logging.info(f"   - Refrigerador: {refrigerator}")

        # Crear DataFrame con las columnas EXACTAS que espera el scaler
        # Basándome en tu imagen, las columnas son: MonthlyHours, Fan, Television, TariffRate, Monitor, Refrigerator
        input_data = pd.DataFrame([[
            monthly_hours, fan, television, tariff_rate, monitor, refrigerator
        ]], columns=['MonthlyHours', 'Fan', 'Television', 'TariffRate', 'Monitor', 'Refrigerator'])

        logging.info(f"🔢 DataFrame creado: {input_data.to_dict('records')[0]}")
        logging.info(f"🏷️ Columnas del DataFrame: {list(input_data.columns)}")
        logging.info(f"📐 Forma del DataFrame: {input_data.shape}")

        # Verificar que las columnas coincidan
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
            actual_features = list(input_data.columns)
            logging.info(f"🔍 Características esperadas: {expected_features}")
            logging.info(f"🔍 Características actuales: {actual_features}")
            
            if expected_features != actual_features:
                logging.warning("⚠️ Las características no coinciden, reordenando...")
                input_data = input_data[expected_features]

        # Escalar datos
        logging.info(f"📊 Datos antes del escalado: {input_data.values[0]}")
        input_scaled = scaler.transform(input_data)
        logging.info(f"⚖️ Datos después del escalado: {input_scaled[0]}")
        
        # Verificar que el escalado funcionó
        if np.array_equal(input_data.values[0], input_scaled[0]):
            logging.error("❌ ERROR: Los datos no se escalaron correctamente!")
            return jsonify({
                'success': False,
                'error': 'Problema con el escalado de datos'
            })

        # Predecir
        prediction = model.predict(input_scaled)[0]
        logging.info(f"🎯 Predicción realizada: {prediction}")

        # Devolver JSON exitoso
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })

    except KeyError as ke:
        logging.error(f"❌ Campo faltante en el formulario: {str(ke)}")
        logging.error(f"📋 Campos disponibles: {list(request.form.keys())}")
        return jsonify({
            'success': False,
            'error': f'Campo faltante: {str(ke)}'
        })
    except ValueError as ve:
        logging.error(f"❌ Error de valor: {str(ve)}")
        logging.error(f"📋 Valores recibidos: {dict(request.form)}")
        return jsonify({
            'success': False,
            'error': 'Valores inválidos. Verifica que todos los campos sean números.'
        })
    except Exception as e:
        logging.error(f"❌ Error en la predicción: {str(e)}")
        logging.error(f"📋 Tipo de error: {type(e).__name__}")
        return jsonify({
            'success': False,
            'error': f'Error en la predicción: {str(e)}'
        })

# Bloquear peticiones no deseadas SILENCIOSAMENTE
@app.before_request
def block_unwanted():
    if request.path.startswith('/logs/'):
        return '', 404

if __name__ == '__main__':
    logging.info("🚀 Iniciando servidor Flask...")
    app.run(debug=True)
