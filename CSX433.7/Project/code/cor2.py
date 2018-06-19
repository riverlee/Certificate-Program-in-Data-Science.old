    ## Compute correlations of columns of a dataframe of mixed types
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


import scipy.stats as ss

def cramers_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))




def corMix(df):
    
    
    def cor_fun(pos_1,pos_2):
        r = 0
        
        ## both numeric
        if(np.issubdtype(df[pos_1].dtype,np.number) and np.issubdtype(df[pos_2].dtype,np.number)):
            
            r = np.corrcoef(df[pos_1],df[pos_2])[0,1]
        
        ## one is numeric and other is a factor/character
        if(np.issubdtype(df[pos_1].dtype,np.number) and not np.issubdtype(df[pos_2].dtype,np.number)):
            regr = linear_model.LinearRegression()

            # Train the model
            regr.fit(np.array(df[pos_1])[:,np.newaxis],df[pos_2],  )
            regr.fit(df.loc[:,pos_1][:,np.new], df.loc[:,pos_2])
       