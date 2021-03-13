import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


room_ix = 3
bedrooms_ix = 4
population_ix = 5
household_ix = 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        rooms_per_household = X[:, population_ix] / X[:, household_ix]
        population_per_house_hold = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_house_hold, bedrooms_per_room]

        return np.c_[X, rooms_per_household, population_per_house_hold]
