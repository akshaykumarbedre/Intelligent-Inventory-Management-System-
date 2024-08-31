from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            national_inv=float(request.form.get('national_inv')),
            lead_time = float(request.form.get('lead_time')),
            in_transit_qty = float(request.form.get('in_transit_qty')),
            forecast_3_month = float(request.form.get('forecast_3_month')),
            sales_1_month = float(request.form.get('sales_1_month')),
            min_bank = float(request.form.get('min_bank')),
            perf_6_month_avg = float(request.form.get('perf_6_month_avg')),

        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        if(pred[0]==1):
            results="The Product Went on Backorder."
        else:
            results="The Product  NOT Went on Backorder."

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)