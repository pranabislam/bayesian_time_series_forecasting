from abc import ABC

from abc import ABC
from abc import abstractmethod
from typing import Literal, Dict, List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import gaussian_kde
import datetime

## Writing skeleton with abstract classes first then will inherit from them afterward

class Distribution(ABC):
    @abstractmethod
    def __init__(self, parameters: dict, num_observations: int = 0):
        self.parameters = parameters
        self.num_observations = num_observations
        
    def __repr__(self):
        data = [f"{param_key} = {param_value}" for param_key, param_value in self.parameters.items()]
        return ", ".join(data)
    
    @property
    def parameters(self) -> Dict[str, float]:
        """
        The parameters attribute of a distribution is a dictionary
        with keys mapping a parameter name to a value
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: Dict[str, float]):
        self._parameters = parameters
    
    @abstractmethod
    def sample(self):
        """
        The sample method should draw random samples from the distribution
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_mean(self) -> float:
        """
        This method should return the expected value of the distribution
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_percentile(self, p) -> float:
        """
        Given a value between 0 and 100, this should return the percentile of the distribution
        """
        raise NotImplementedError

        
class Model(ABC):
    @abstractmethod
    def __init__(self, model_parameters: dict):
        self.model_parameters = {}
    
    @abstractmethod
    def train(self, data: pd.DataFrame, start_date: str, end_date: str) -> None:
        """
        The train method should use the training data provided in data.txt to 
        update the model parameters that can be used to generate predictions 
        """
        raise NotImplementedError
        
    @abstractmethod
    def predict(self, dates: List[str], type_: Literal["A", "B", "C", "TOTAL"]) -> Dict[str, Distribution]:
        """
        Given a set of dates to predict, this method should return a dictionary
        with keys at the date (YYYYmmdd) and a Distribution type object representing
        the distributional forecast in the future.  The type_ input, specifies which 
        product type to generate the prediction for (TOTAL should be the sum)
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, dates: List[str], type_: Literal["A", "B", "C", "TOTAL"]) -> Dict[str, int]:
        """
        Given a set of dates to sample, this method should return a dictionary
        with keys at the date (YYYYmmdd) and a sample from the distributional forecast.
        The type_ input, specifies which 
        product type to generate the samples for (TOTAL should be the sum)
        """
        raise NotImplementedError

class myDistribution(Distribution):

    ## override base class to fit empirical distbns
    def __init__(self, empiricalPreds: np.ndarray):
        self.empiricalPreds = empiricalPreds
        self.sample_pdf = gaussian_kde(self.empiricalPreds)
        self.new_sample_data = self.sample_pdf.resample(10000).T[:,0]
        
        
    def __repr__(self):
        return f"Gaussian kernel fitted to empirical distribution used. Mean (point prediction): {self.get_mean()}"
    
    @property
    def parameters(self) -> Dict[str, float]:
        """
        The parameters attribute of a distribution is a dictionary
        with keys mapping a parameter name to a value
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: Dict[str, float]):
        self._parameters = parameters
    
    def sample(self, n: int = 1):
        """
        The sample method should draw random samples from the distribution
        """
        if n == 1:
            return self.sample_pdf.resample(n)[0][0]
        return self.sample_pdf.resample(n)[0]
    
    def get_mean(self) -> float:
        """
        This method should return the expected value of the distribution
        """
        ## We assign emp mean as mean to match the predict method of rf
        ## This will misalign slightly with the fitted gaussian kde
        mean =  np.mean(self.empiricalPreds) 
        return mean
        
    def get_percentile(self, p) -> float:
        """
        Given a value between 0 and 100, this should return the percentile of the distribution
        """
        return np.percentile(self.new_sample_data, p)

class myModel(Model):
    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)
        self.forecastPeriods = 22 ## total days to forecast
        self.models = {
            'A': [RandomForestRegressor(**model_parameters) for _ in range(self.forecastPeriods)],
            'B': [RandomForestRegressor(**model_parameters) for _ in range(self.forecastPeriods)],
            'C': [RandomForestRegressor(**model_parameters) for _ in range(self.forecastPeriods)],
        }
        self.lastDay = None ## fill in when you make features

    def preprocessAndMakeFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        train_A = data[data.TYPE == 'A']
        train_B = data[data.TYPE == 'B']
        train_C = data[data.TYPE == 'C']
        
        train = pd.concat(
            [
                train_A.rename(columns={'VOLUME':'A'})[['A']],
                train_B.rename(columns={'VOLUME':'B'})[['B']],
                train_C.rename(columns={'VOLUME':'C'})[['C']],
                train_A.index.isocalendar(), ## All products share same index; this is fine
            ],
        axis = 1,
        )
        
        ## Moving average
        for col in ['A','B','C']:
            for window in [15,30,45,60,75]:
                roll_col_ma = f"{col}_{window}_day_ma"
                train[roll_col_ma] = train[col].rolling(window).mean()

        ## Volume ratios between products over lag periods
        for window in [15,45,75]:
            train[f'vol_ratio_AB_{window}_days'] = train.A.rolling(window).sum() / train.B.rolling(window).sum()
            train[f'vol_ratio_BC_{window}_days'] = train.B.rolling(window).sum() / train.C.rolling(window).sum()
            train[f'vol_ratio_AC_{window}_days'] = train.A.rolling(window).sum() / train.C.rolling(window).sum()

        ## Lags for each volume up to 5 days
        for col in ['A','B','C']:
            for window in range(1,6):
                roll_col_lag = f"{col}_{window}_day_lag"
                train[roll_col_lag] = train[col].shift(window)
        
        self.lastDay = train.iloc[-1,:].to_frame().T

        return train
    
    def trainOneModel(
            self,
            data: pd.DataFrame, 
            forecastPeriod: int, 
            type_: str,
        ):
        '''
        Given a model, return model trained to predict forecastPeriod days into the future for a specific product.
        Forecast period of 1 implies we use the first model we saved
        '''
        model = self.models[type_][forecastPeriod - 1]
        label_name = f'label_{type_}_{forecastPeriod}_day_future'
        label = data[type_].shift(-1 * forecastPeriod).rename(label_name)
        data = data.join(label).dropna()
        print(f"Training from {data.index.min()} through {data.index.max()} for {forecastPeriod} days in future for {type_}")
        model.fit(
            X=data.drop(columns=label_name),
            y=data[label_name],
        )
        return model
    
    def train(self, data: pd.DataFrame, start_date: str, end_date: str) -> None:
        """
        The train method should use the training data provided in data.txt to 
        update the model parameters that can be used to generate predictions 
        """
        ## WILL ADD FUNCTIONALITY FOR TRAIN START AND TRAIN END LATER

        data = self.preprocessAndMakeFeatures(data)
        for type_ in ['A','B','C']:
            for forecastPeriod in range(1, self.forecastPeriods + 1):
                self.models[type_][forecastPeriod - 1] = self.trainOneModel(
                    data, 
                    forecastPeriod, 
                    type_
                )
        
    def calcTotal(self, dates: List[str]) -> Dict[str, Distribution]:
        '''
        Helper function to calculate total of A B C 
        '''
        distributions = []
        ret = {dateString:None for dateString in dates}
        for type_ in ['A','B','C']:
            distributions.append(self.predict(dates, type_))
        
        for subDict in distributions:
            for dateString, partialDistribution in subDict.items():
                if ret[dateString] is None:
                    ret[dateString] = partialDistribution.empiricalPreds
                else:
                    ret[dateString] += partialDistribution.empiricalPreds
        ret = {dateString:myDistribution(empiricalPreds) for dateString, empiricalPreds in ret.items()}
        return ret
        
    def predict(self, dates: List[str], type_: Literal["A", "B", "C", "TOTAL"]) -> Dict[str, Distribution]:
        """
        Given a set of dates to predict, this method returns a dictionary
        with keys at the date (YYYYmmdd) and a Distribution type object representing
        the distributional forecast in the future.  The type_ input, specifies which 
        product type to generate the prediction for (TOTAL is for the sum)
        """
        if type_ == 'TOTAL':
            return self.calcTotal(dates)
        
        forecastPeriods = [((datetime.datetime.strptime(dateString, '%Y%m%d') - 
                            datetime.datetime.strptime('20220901', '%Y%m%d')).days,
                            dateString)
            for dateString in dates
        ]
        ret = {}
        for forecastPeriod, dateString in forecastPeriods:
            currModel = self.models[type_][forecastPeriod - 1]
            preds = np.zeros(len(currModel.estimators_))
            
            for i, pred in enumerate(currModel.estimators_):
                preds[i] = pred.predict(self.lastDay)[0]
            
            distbn = myDistribution(preds)

            ret[dateString] = distbn
        
        return ret
        
    
    def sample(self, dates: List[str], type_: Literal["A", "B", "C", "TOTAL"]) -> Dict[str, int]:
        """
        Given a set of dates to sample, this method returns a dictionary
        with keys at the date (YYYYmmdd) and a sample from the distributional forecast.
        The type_ input, specifies which 
        product type to generate the samples for (TOTAL is for the sum)
        """
        predsWithDistributions = self.predict(dates, type_)
        return {dateString:distbn.sample() for dateString, distbn in predsWithDistributions.items()}
    
