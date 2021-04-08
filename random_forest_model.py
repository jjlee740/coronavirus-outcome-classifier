import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

def random_forest_model():
    p = Path('.')
    dirs = [x for x in p.iterdir() if x.is_dir()]
    f = str(dirs[-1])+ '/cases_train_processed.csv'

    # Read CSV file, drop unneeded columns, factorize the remaining ones 
    df = pd.read_csv(f, dtype={"additional_information": "string", "source": "string"})
    df = df.drop(columns=['source', 'Last_Update','additional_information', 'province', 'country', 'Province_State', 'Combined_Key'])
    df = df.apply(lambda x: pd.factorize(x)[0])

    train, test =  train_test_split(df, test_size=0.2, random_state=0)

    # extract the name of the features
    features = df.columns[:15]

    # extract the outcome
    y = pd.factorize(train['outcome'])[0]

    # build classifier
    clf = RandomForestClassifier(n_jobs=120, random_state=0)
    clf.fit(train[features], y)

    # use classifier to build predictions
    preds = clf.predict(test[features])

    # output results
    pd.crosstab(test['outcome'], preds, rownames=['Actual Outcome'], colnames=['Predicted Outcome'])
    list(zip(train[features], clf.feature_importances_))

    # save classifier as a .pkl file
    filename = str(dirs[1])+'/randomforest_classifier.pkl'
    pickle.dump(clf, open(filename, 'wb'))

if __name__ == "__main__":
	random_forest_model()
