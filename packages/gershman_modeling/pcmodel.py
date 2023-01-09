# Code for running and fitting the policy compression model

import numpy as np
import statsmodels.api as sm


def fit_marginal(data, K = 50):
    # Fit marginal choice probability
    #
    # INPUTS:
    #   data - dataframe containing all the relevant data
    #   K - number of different exponential weighting values
    #   
    # OUTPUTS:
    #   data with new column 'logodds' (choice log odds)

    alpha = np.linspace(0.001,0.3,num=K)
    N = data.shape[0]
    m = np.zeros((N,K)) + 0.5
    c = data['Decision']
    sess = data['Session']

    for n in range(N):
        if (n > 0) and (sess.iat[n] == sess.iat[n-1]):
            m[n,] = (1-alpha)*m[n-1,] + alpha*c.iat[n-1]

    m[m==0] = 0.001
    m[m==1] = 0.999
    L = np.dot(c,np.log(m)) + np.dot((1-c),np.log(1-m))
    m = m[:,np.argmax(L)]
    data['logodds'] = np.log(m) - np.log(1-m)
    print("alpha =",alpha[np.argmax(L)])

    return data


def sim(data, p = 0.8, q = 0.98):
    # Simulate model
    #
    # INPUTS:
    #   data - dataframe containing all the relevant data
    #   p - probability of reward for correct action
    #   q - state stay probability
    #   
    # OUTPUTS:
    #   data with new columns 'qdiff' (Q-value differences) and 'rpe' (reward prediction error)
    
    N = data.shape[0]
    qdiff = np.zeros(N)
    rpe = np.zeros(N)
    
    c = data['Decision']
    sess = data['Session']
    r = data['Reward']
    
    for n in range(N):
        
        if (n == 0) or (sess.iat[n] != sess.iat[n-1]):
            b = 0.5
        
        if c.iat[n]==1:
            P = p
            rpe[n] = r.iat[n] - b*p - (1-b)*(1-p)
        else:
            P = 1-p
            rpe[n] = r.iat[n] - b*(1-p) - (1-b)*p
        
        if r.iat[n]==0:
            P = 1-P
        
        # compute value difference
        qdiff[n] = (2*b-1)*(2*p-1)
        
        # update belief
        b = (q*b*P + (1-q)*(1-b)*(1-P))/(b*P + (1-b)*(1-P))
    
    data['qdiff'] = qdiff
    data['rpe'] = rpe
    
    return data


def crossval(data, m):
    # Cross-validation
    # INPUTS:
    #   data - dataframe with all relevant data
    #   m - model (0 = only qdiff; 1 = qdiff + logodds, fitted; 2 = qdiff + fixed coefficient for logodds)
    #
    # OUTPUTS:
    #   loglik - log likelihood of held-out data
    
    traindata = data.sample(frac=0.7, random_state=1)
    testdata = data.drop(traindata.index)
    
    results = fitglm(traindata, m)
    y_test,X_test,z_test = createvars(testdata,m)
    y_pred = results.predict(X_test,offset=z_test)
    y_pred[y_pred==0] = 0.001
    y_pred[y_pred==1] = 0.999
    loglik = np.sum(y_test*np.log(y_pred) + (1-y_test)*np.log(1-y_pred))
    
    return loglik


def fitglm(data, m):
    # Fit GLM with maximum likelihood estimation
    
    y,X,z = createvars(data,m)
    model = sm.GLM(y,X,family=sm.families.Binomial(),offset=z)
    results = model.fit()
    
    return results


def createvars(data, m):
    # Create variables for GLM
    
    y = data['Decision']
    
    if m == 0:
        X = data['qdiff']
        z = np.zeros(X.shape[0])
    elif m == 1:
        X = data[['qdiff','logodds']]
        z = np.zeros(X.shape[0])
    elif m == 2:
        X = data['qdiff']
        z = data['logodds']
        
    return y,X,z


def fit(data):
    # Fit model to all subjects and get cross-validation results
    #
    # INPUTS:
    #   data - dataframe with all relevant data
    #
    # OUTPUTS:
    #   bic - subjects x model BIC values
    #   loglik - subjects x model log likelihood values from cross-validation
    #   param - model parameters for each subject for model 1 (qdiff + logodds)
    #   data - dataframe with 'qdiff' (Q-value difference),'rpe' (reward prediction error)
    #          and 'logodds' (log choice probability) columns added
    
    subjects = data['Mouse'].unique()
    S = len(subjects)
    bic = np.zeros((S,3))
    loglik = np.zeros((S,3))
    param = np.zeros((S,2))
    data = sim(data)
    data = fit_marginal(data)

    for s in range(S):
        df = data[data['Mouse']==subjects[s]]
        for m in range(3):
            results = fitglm(df, m)
            bic[s,m] = results.bic_llf
            loglik[s,m] = crossval(df, m)
            if m==2:
                param[s,] = results.params
    
    return bic, loglik, param, data
