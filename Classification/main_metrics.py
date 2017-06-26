import pandas as pd
import numpy as np

def main_metrics(conf_matrix,labels):
    '''Returns a dataframe with the main metrics
    Args:
        conf_matrix: A confusion matrix such as the returned using NLTK
        labels: The labels of the lassifier
    Returns:
        dataframe with the metrics
    '''
    index_df=[]
    FP = (conf_matrix.sum(axis=0) - np.diag(conf_matrix)).astype('float')
    dictionary={labels[0]:FP[0],labels[1]:FP[1],labels[2]:FP[2]}
    df=pd.DataFrame(dictionary,index=[0])
    index_df.append("FP")
    
    FN = (conf_matrix.sum(axis=1) - np.diag(conf_matrix)).astype('float')
    dictionary={labels[0]:FN[0],labels[1]:FN[1],labels[2]:FN[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("FN")
    
    TP = (np.diag(conf_matrix)).astype('float')
    dictionary={labels[0]:TP[0],labels[1]:TP[1],labels[2]:TP[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("TP")
    
    TN = (conf_matrix.sum() - (FP + FN + TP)).astype('float')
    dictionary={labels[0]:TN[0],labels[1]:TN[1],labels[2]:TN[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("TN")
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    dictionary={labels[0]:TPR[0],labels[1]:TPR[1],labels[2]:TPR[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("TPR")
    
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    dictionary={labels[0]:TNR[0],labels[1]:TNR[1],labels[2]:TNR[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("TNR")
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    dictionary={labels[0]:PPV[0],labels[1]:PPV[1],labels[2]:PPV[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("PPV")
    
    # Negative predictive value
    NPV = TN/(TN+FN)
    dictionary={labels[0]:NPV[0],labels[1]:NPV[1],labels[2]:NPV[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("NPV")
    
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    dictionary={labels[0]:FPR[0],labels[1]:FPR[1],labels[2]:FPR[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("FPR")
    # False negative rate
    FNR = FN/(TP+FN)
    dictionary={labels[0]:FNR[0],labels[1]:FNR[1],labels[2]:FNR[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("FNR")
    
    # False discovery rate
    FDR = FP/(TP+FP)
    dictionary={labels[0]:FDR[0],labels[1]:FDR[1],labels[2]:FDR[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("FDR")
    
    # Overall accuracy
    ACC = ((TP+TN)/(TP+FP+FN+TN))
    dictionary={labels[0]:ACC[0],labels[1]:ACC[1],labels[2]:ACC[2]}
    df_aux=pd.DataFrame(dictionary,index=[0])
    df=df.append(df_aux)
    index_df.append("ACC")
    df.index=(index_df)
    
    return df
    