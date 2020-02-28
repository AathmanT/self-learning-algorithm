# import sys
# sys.path.append('/content/pygsom/')
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
import gsmote.preprocessing as pp
from gsmote import GeometricSMOTE
from GSOM import GSOM
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from gsmote.comparison_testing.Evaluator import evaluate
import sys

sys.path.append('../../')

class MeanClassifier(BaseEstimator,ClassifierMixin):
    """An example of classifier"""

    def __init__(
            self,
            smooth_iteration=25,
            training_iteration=50,
            spreading_factor = 0.83,
            FD=0.1,
            learning_rate=0.3,
            smooth_learning_factor=0.8
    ):
        """
        Called when initializing the classifier
        """
        self.smooth_iteration = smooth_iteration
        self.spreading_factor =spreading_factor
        self.training_iteration = training_iteration
        self.FD = FD
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.gsom = GSOM(self.spreading_factor,55, max_radius=4, FD=self.FD,learning_rate=self.learning_rate,
                         smooth_learning_factor=self.smooth_learning_factor)
        self.gsmote = GeometricSMOTE(random_state=1, truncation_factor=1.0,
                                     deformation_factor=0, k_neighbors=5,
                                     sampling_rate= 0.3)

    def fit(self, X, y):
        X_train, y_train = self.gsmote.fit_resample(X, y)
        y1 = np.copy(y_train)
        y = np.column_stack([y1, y_train])
        labels = ["Name", "label"]
        y = np.vstack((labels, y))
        frame = pd.DataFrame(y[1:, :], columns=y[0, :])
        self.gsom.fit(X_train,self.training_iteration,self.smooth_iteration)
        self.gsom.labelling_gsom(X_train, frame, "Name", "label")
        self.gsom.finalize_gsom_label()
        return self


    # def _meaning(self, x):
    #     return True

    def predict(self, X):
        return self.gsom.predict_values(X)

    # def score(self, X, y=None):
    #     # counts number of values bigger than mean
    #     return(sum(self.predict(X)))
    #

from sklearn.model_selection import GridSearchCV, train_test_split

date_file = "../../data/adultmini.csv".replace('\\','/')
# date_file = "content/pygsom/data/adult.csv".replace('\\','/')

X,y = pp.preProcess(date_file)

# try different combination of hyper paramenters
parameters = [{'smooth_iteration':[12,25],'training_iteration':[25,50],'spreading_factor':[0.83,0.53,0.7],'FD':[0.1,0.05,0.2],
               'learning_rate':[0.3,0.4,0.5],'smooth_learning_factor':[0.8,0.6,0.7]}]
gs = GridSearchCV(MeanClassifier(), parameters)
gs.fit(X,y)

params = gs.best_params_
print (params)
#
# #find performance
# X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# gsmote = GeometricSMOTE(random_state=1, truncation_factor=params["truncation_factor"],
#                                      deformation_factor=params["deformation_factor"], k_neighbors=params["k_neighbors"],
#                                      sampling_rate=params["sampling_rate"])
# X_train,y_train = gsmote.fit_resample(X_t,y_t)
# # Fitting Gradient boosting
# gbc = GradientBoostingClassifier (n_estimators=params["n_estimators"], learning_rate = params["learning_rate"],
#                                   max_depth = params["max_depth"])
# gbc.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_predict = gbc.predict(X_test)
# y_pred = np.where(y_predict.astype(int)>0.5,1,0)
#
# evaluate("Gradient Boosting",y_test,y_pred)