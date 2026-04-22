import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.preprocessor = load_object("artifacts/preprocessor.pkl")  # feature_engine + encoder
        self.model        = load_object("artifacts/model.pkl")

    def predict(self, input_df: pd.DataFrame) -> float:
        try:
            X          = self.preprocessor.transform(input_df)  # feature_engine → encoder
            prediction = self.model.predict(X)
            return float(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Collects raw user inputs from Streamlit and builds a DataFrame
       with exactly the same columns FeatureEngine saw during training."""

    def __init__(
        self,
        manufacturer     : str,
        category         : str,
        fuel_type        : str,
        gear_box_type    : str,
        color            : str,
        leather_interior : str,
        turbo            : str,
        mileage          : int,
        engine_volume    : float,
        age              : int,
        airbags          : int,
        levy             : float,
        is_premium_brand : bool,
        inventory_segment: str,
    ):
        self.manufacturer      = manufacturer
        self.category          = category
        self.fuel_type         = fuel_type
        self.gear_box_type     = gear_box_type
        self.color             = color
        self.leather_interior  = leather_interior
        self.turbo             = turbo
        self.mileage           = mileage
        self.engine_volume     = engine_volume
        self.age               = age
        self.airbags           = airbags
        self.levy              = levy
        self.is_premium_brand  = is_premium_brand
        self.inventory_segment = inventory_segment

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Returns a single-row DataFrame matching training column structure."""
        return pd.DataFrame({
            # ── Raw columns FeatureEngine expects ──────────────────────
            'Manufacturer'     : [self.manufacturer],
            'Category'         : [self.category],
            'Fuel type'        : [self.fuel_type],
            'Gear box type'    : [self.gear_box_type],
            'Color'            : [self.color],
            'Leather interior' : [self.leather_interior],
            'Turbo'            : [self.turbo],
            'Mileage'          : [self.mileage],
            'Engine_Volume_Num': [self.engine_volume],
            'Engine volume'    : [self.engine_volume],
            'Age'              : [self.age],
            'Airbags'          : [self.airbags],
            'Levy'             : [self.levy],
            'Is_Premium_Brand' : [self.is_premium_brand],
            'Inventory_Segment': [self.inventory_segment],
            'Prod. year'       : [2025 - self.age],
            'Cylinders'        : [4],
            'Wheel'            : ['Left wheel'],
            'Doors'            : ['4-May'],
            'Model'            : ['Unknown'],
            'Price_Per_Litre'  : [0],
            'Age_Group'        : ['Unknown'],
            'Mileage_Band'     : ['Unknown'],
        })