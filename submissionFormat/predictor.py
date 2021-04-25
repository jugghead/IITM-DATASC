### Custom definitions and classes if any ###
import pandas as pd
import numpy as np 
import joblib

def predictRuns(input_test):
    prediction = 0
    ### Your Code Here ###

    with open('regression_model.joblib','rb') as f:
        regressor=joblib.load(f)
    with open('venue_encoder.joblib','rb') as f:
        venue_encoder=joblib.load(f)
    with open('team_encoder.joblib','rb') as f:
        team_encoder=joblib.load(f)

    # read test data
    test_case = pd.read_csv(input_test)
    # encode venue and batting and bowling teams
    test_case['venue']        =venue_encoder.transform(test_case['venue'])
    test_case['batting_team'] =team_encoder.transform(test_case['batting_team'])
    test_case['bowling_team'] =team_encoder.transform(test_case['bowling_team'])
    
    # make sure that the order of the columns is same
    test_case = test_case[['venue','innings','batting_team','bowling_team']]
    
    # convert input test case into numpy array
    testArray = test_case.to_numpy()

    # one hot encode venue,batting and bowling teams
    test_case = np.concatenate((np.eye(42)[testArray[:,0]],
                                np.eye(2)[testArray[:,1]-1],
                                np.eye(15)[testArray[:,2]],
                                np.eye(15)[testArray[:,3]],),
                                axis = 1)
    prediction=regressor.predict(test_case)
    return prediction
