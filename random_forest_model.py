from pyspark.ml.regression import RandomForestRegressor as RFR
from keystone.models.base_model import BaseModel

class RFRegressionModel(BaseModel):
    def load(self, path):
	return RFR.load(path)

	    
#     def fit(self, dataframe):
#         for f in self._features:
# 	    dataframe = f.transform(dataframe)
#         dataframe = self._features_assembler.transform(dataframe)
#         #return self._model.fit(dataframe)
#         self._model = self._model.fit(dataframe)
#         return self
#      