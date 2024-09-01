from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class Training_Pipeline:
    def __init__(self, path=None):
        self.data_ingestion_obj = Dataingestion(path)
        self.data_transform_obj = DataTransformation()
        self.model_trainer_obj=ModelTrainer()

    def initiate_training_pipeline(self):
        train_path, test_path = self.data_ingestion_obj.initiate_data_ingestion()
        train_data, test_data, preprocesser = self.data_transform_obj.initiate_data_transformation(
            train_path, test_path)
        result=self.model_trainer_obj.initate_model_training(train_data, test_data)
        return result

