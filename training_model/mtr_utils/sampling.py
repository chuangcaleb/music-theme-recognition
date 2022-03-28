from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def oversample(x, y, random_stae):
    """ Oversampling with RandomOverSampler """

    ros = RandomOverSampler(random_state=random_stae, shrinkage=0.4)
    return ros.fit_resample(x, y)


def undersample(x, y, random_state):
    """ Undersampling with RandomUnderSampler """

    rus = RandomUnderSampler(random_state=0)
    return rus.fit_resample(x, y)


def smote(x, y, random_state):
    """ Oversampling using SMOTE """

    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(x, y)
