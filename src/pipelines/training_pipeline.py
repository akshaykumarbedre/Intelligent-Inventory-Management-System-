from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import DataTransformation


class Training_Pipeline:
    def __init__(self):
        self.data_ingestion_obj=Dataingestion()
        self.data_transform_obj=DataTransformation()

    def initiate_training_pipeline(self):
        train_path,test_path=self.data_ingestion_obj.initiate_data_ingestion()
        train_data,test_data,preprocesser=self.data_transform_obj.initiate_data_transformation(train_path, test_path)

tp=Training_Pipeline()
tp.initiate_training_pipeline()