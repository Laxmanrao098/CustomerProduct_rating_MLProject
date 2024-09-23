import logging 
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStratergy(ABC):
    """
    Abstract class defining stratergy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStratergy(DataStratergy):
    """
    Stratergy for Preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data =  data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in Preprocessing data: {}".format(e))
            raise e

class DataDividestratergy(DataStratergy):
    """
    Stratergy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        
class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """

    def __init__(self, data: pd.DataFrame, stratergy: DataStratergy): 
        self.data = data
        self.stratergy = stratergy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Handle Data
        """
        try:
            return self.stratergy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e