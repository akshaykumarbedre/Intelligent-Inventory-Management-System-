from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.pipelines.training_pipeline import Training_Pipeline
import os
from src.logger import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info("Accessing index page")
    return render_template('index.html')

@app.route('/train_custom_data', methods=['GET', 'POST'])
def training_pipeline():
    if request.method == 'POST':
        logging.info("Received POST request for training custom data")
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return render_template('train_model.html', error="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            logging.warning("No selected file")
            return render_template('train_model.html', error="No selected file")
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(f"{filepath}")
                logging.info(f"File saved: {filepath}")

                Training_model = Training_Pipeline(filepath)
                result = Training_model.initiate_training_pipeline()
                logging.info("Training pipeline initiated successfully")

                return render_template('train_model.html', success=result)
            except Exception as e:
                logging.error(f"Error during training: {str(e)}", exc_info=True)
                return render_template('train_model.html', error="Follow the template file to train the model")
        else:
            logging.warning(f"Invalid file type: {file.filename}")
            return render_template('train_model.html', error="Invalid file type")
    
    logging.info("Accessing train_model page")
    return render_template('train_model.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("Accessing prediction form")
        return render_template('form.html')
    else:
        logging.info("Received POST request for prediction")
        try:
            data = CustomData(
                national_inv=float(request.form.get('national_inv')),
                lead_time=float(request.form.get('lead_time')),
                in_transit_qty=float(request.form.get('in_transit_qty')),
                forecast_3_month=float(request.form.get('forecast_3_month')),
                sales_1_month=float(request.form.get('sales_1_month')),
                min_bank=float(request.form.get('min_bank')),
                perf_6_month_avg=float(request.form.get('perf_6_month_avg')),
            )
            
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            
            results = round(pred[0], 2)
            if pred[0] == 1:
                results = "The Product Went on Backorder."
            else:
                results = "The Product NOT Went on Backorder."
            logging.info(f"Prediction result: {results}")
            return render_template('form.html', final_result=results)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}", exc_info=True)
            return render_template('form.html', error="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    logging.info("Starting the Flask application")
    app.run(host='0.0.0.0', debug=True, port=5000)