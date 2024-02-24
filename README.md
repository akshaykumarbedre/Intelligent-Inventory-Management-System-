**Backorder Prediction Model**

This project is about predicting whether a product will go on backorder. The model is trained on various features like national inventory, lead time, in transit quantity, forecast sales for the next 3 months, sales quantity for the prior 1 month, minimum recommended stock level, and 6-month average performance.

**Project Structure**

The project is divided into several Python scripts each serving a specific purpose:

1. logger.py: This script sets up the logging configuration.
2. utils.py: This script contains utility functions for saving and loading objects, removing outliers from a dataframe, and encoding the target column.
3. exception.py: This script defines a custom exception class for handling exceptions throughout the project.
4. data_ingestion.py: This script is responsible for ingesting the data from CSV files.
5. data_transformation.py: This script is responsible for preprocessing the data, including handling missing values, scaling numerical features, and encoding categorical features.
6. model_trainer.py: This script is responsible for training the model. It includes functions for evaluating multiple models and selecting the best one based on F1 score.
7. prediction_pipeline.py: This script is responsible for making predictions on new data using the trained model.
8. training_pipeline.py: This script initiates the training pipeline which includes data ingestion, data transformation, and model training.

**Usage**

To train the model, create an instance of the Training_Pipeline class and call the initiate_training_pipeline method. This will ingest the data, transform it, and train the model.

tp = Training_Pipeline()

tp.initiate_training_pipeline()

To make predictions on new data, create an instance of the CustomData class with the feature values, convert it to a dataframe, create an instance of the PredictPipeline class, and call the predict method.

cd = CustomData(national_inv=24, lead_time=8, in_transit_qty=0, forecast_3_month=3456, sales_1_month=10, min_bank=7, perf_6_month_avg=1)

df = cd.get_data_as_dataframe()

pp = PredictPipeline()

print(pp.predict(df))

**Note**

Please ensure that all the necessary data files are in the correct paths as specified in the scripts. Also, make sure to handle any exceptions that may occur during the execution of the scripts. The CustomException class can be used for this purpose.

This project is a basic implementation and can be further improved by fine-tuning the models, adding more features, or using more advanced models. Always validate the model with new data to ensure its effectiveness.

Happy coding!
