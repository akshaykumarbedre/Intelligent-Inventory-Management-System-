**<h1>Backorder Prediction Model</h1>**

This project is about predicting whether a product will go on backorder. The model is trained on various features like national inventory, lead time, in transit quantity, forecast sales for the next 3 months, sales quantity for the prior 1 month, minimum recommended stock level, and 6-month average performance.

<h2>project Structure</h2>

The project is divided into several Python scripts each serving a specific purpose:

1. logger.py: This script sets up the logging configuration.
2. utils.py: This script contains utility functions for saving and loading objects, removing outliers from a dataframe, and encoding the target column.
3. exception.py: This script defines a custom exception class for handling exceptions throughout the project.
4. data_ingestion.py: This script is responsible for ingesting the data from CSV files.
5. data_transformation.py: This script is responsible for preprocessing the data, including handling missing values, scaling numerical features, and encoding categorical features.
6. model_trainer.py: This script is responsible for training the model. It includes functions for evaluating multiple models and selecting the best one based on F1 score.
7. prediction_pipeline.py: This script is responsible for making predictions on new data using the trained model.
8. training_pipeline.py: This script initiates the training pipeline which includes data ingestion, data transformation, and model training.

<h2>How to run this app?</h2>

Code is written in Python 3.7. If you don't have python installed on your system, click here https://www.python.org/downloads/ to install.<br>
Create virtual environment: conda create -n myenv python=3.7.<br>
Activate the environment: conda activate myenv.<br>
Install the packages: pip install -r requirements.txt<br>
Run the app: python application.py 

<h2> Deployment Model in AWS</h2>

Our application is deployed on AWS (Amazon Web Services), leveraging its robust, scalable, and reliable infrastructure.<br>
Backorder prediction projest link :  [http://backorder1-env.eba-u6qs2uzh.ap-southeast-2.elasticbeanstalk.com](http://backorderprediction-env.eba-3ivrmujp.ap-south-1.elasticbeanstalk.com),  
The link is currently unavailable due to privacy settings on the project
**Note**

Please ensure that all the necessary data files are in the correct paths as specified in the scripts. Also, make sure to handle any exceptions that may occur during the execution of the scripts. The CustomException class can be used for this purpose.

This project is a basic implementation and can be further improved by fine-tuning the models, adding more features, or using more advanced models. Always validate the model with new data to ensure its effectiveness.

Happy coding!
