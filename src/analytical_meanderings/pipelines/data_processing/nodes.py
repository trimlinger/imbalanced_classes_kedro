"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import pandas as pd
import numpy as np
import random
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

def preprocess_dmf(dmf: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess dmf_data
    
    Args:
        dmf: raw data
        
    Returns:
        Preprocessed data; null values in 'therapeutic_class' removed
    """
    dmf=dmf[dmf.therapeutic_class.notnull()]
    return dmf

#def run_imbalanced_classes(raw_shortage_prediction: pd.DataFrame):
def run_imbalanced_classes(raw_shortage_prediction):
    """  Run imbalanced_classes.py  """
    
    model_cols = ['Minority_Portion', 'Recall', 'Precision', 'Accuracy']
    zeros = pd.Series([0]*3)
    accuracy_str = 'Overall accuracy is {:.1%}'
    recall_str = 'Recall: {} positive (default) cases, {} ({:.1%})' +\
        ' correct predictions'
    precision_str = 'Precision: {} positive (default) predictions,' +\
        ' {} ({:.1%}) true positives'

    ###########################################################################
    # This function runs the neural network to evaluate all accuracy metrics  
    ###########################################################################
    def run_model(X_train, X_test, y_train, y_test, i_run, verbose=False):
        
        if verbose:
            print('Run number: {}'.format(i_run))
            print('______________')
            print(y_train.value_counts())

        # Scale data with StandardScaler
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # This model uses the MLPClassifier neural network
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic',
                            max_iter=2000, random_state=42)

        # Fit model using training data
        mlp.fit(X_train, y_train)

        # Get model predictions
        predictions = mlp.predict(X_test)

        # Calculate number of true/false positives, true/false negatives, and
        # other related info
        N_all = y_test.size
        N_actualPos = np.count_nonzero(y_test)
        N_actualNeg = N_all - N_actualPos

        tp_sum_counts = (y_test + predictions).value_counts().sort_index()

        N_trueNeg, N_false, N_truePos = tp_sum_counts.add(zeros, fill_value=0)
        N_falseNeg = N_actualPos - N_truePos
        N_true = N_all - N_false
        N_falsePos = N_false - N_falseNeg
        N_allPos = N_truePos + N_falsePos

        # Calculate model accuracy
        accuracy = N_true / N_all
        if verbose:
            print(accuracy_str.format(accuracy))

        # Calculate model recall
        recall = N_truePos / N_actualPos
        if verbose:
            print(recall_str.format(N_actualPos, N_truePos, recall))

        # Calculate model precision, and exit if undefined
        try:
            precision = N_truePos / N_allPos
            if verbose:
                print(precision_str.format(N_allPos, N_truePos, precision))
        except:
            print('No default predictions'); exit()

        if verbose:
            print('')
        return accuracy, recall, precision

    ###########################################################################
    # Create an interactive graph using the plotly.express package
    ###########################################################################
    def interactive_graph(data, algo):
        title = 'Precision Recall Curves for {} Algorithm'.format(algo)
        hover_data = dict(list(zip(model_cols, [':.3f']*len(model_cols))))
        fig = px.scatter(data, x='Recall', y='Precision', height=600,
                         width=600, title=title, color='Minority_Portion',
                         hover_data=hover_data)

        fig.show()
        fig.write_html('./data/02_intermediate/{}.html'.format(algo))

    ###########################################################################
    # Sampling algorithms for oversample, undersample, and smote
    ###########################################################################
    def sample(model_data, i_run, add_n=None, remove_n=None, ratio=None):
        X_train, X_test, y_train, y_test = model_data
        
        # Case of undersampling algo
        if remove_n:
            i_drop = np.random.choice(false_train_index, remove_n,
                                      replace=False)
            X_train = X_train.drop(i_drop)
            y_train = y_train.drop(i_drop)

        # Case of oversampling algo
        elif add_n:
            i_add = true_train.sample(n=add_n, replace=True,
                                      random_state=42).index
            X_train = X_train.append(X_train.loc[i_add], ignore_index=True)
            y_train = y_train.append(y_train.loc[i_add], ignore_index=True)

        # Case of SMOTE algo
        elif ratio:
            sm = SMOTE(sampling_strategy=ratio, random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        minority_portion = y_train[y_train==1].size / y_train.size
        acc, rec, prec = run_model(X_train, X_test, y_train, y_test, i_run)

        return [minority_portion, rec, prec, acc]

    ###########################################################################
    # Script begins here 
    ###########################################################################
    # ETL to clean up the data file
    drugs_formatted = raw_shortage_prediction.copy()

    # Drop monograph_name column
    drugs_formatted = drugs_formatted.drop('monograph_name', axis=1)

    # List of nonbinary columns
    nb_col = ['price_scaled_dosage_form', 'total_inspections', 'age_of_drug',
              'mw']

    # Cast all binary data as integers zeros or ones
    for col in drugs_formatted:
        if col not in nb_col:
            drugs_formatted[col] = drugs_formatted[col].astype(bool)\
                                                       .astype(int)

    # Define target column
    target = 'rolling12_shortage'

    # Define the standard (unsampled) training and testing datasets
    df_copy = drugs_formatted.copy()
    X = df_copy.drop(target, axis=1)
    y = df_copy[target]
    model_data = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = model_data

    # Get some descriptors for the training and testing datasets
    false_train = y_train[y_train==0]
    false_train_index = false_train.index
    false_train_size = false_train.size
    true_train = y_train[y_train==1]

    # Over sampling -- can play around with start and stop
    start = 0
    stop = y_train.size
    add_n_values = np.linspace(start, stop, num=50, dtype=int)
    oversample_data = [sample(model_data, i_run, add_n=add_n)
                       for i_run, add_n in enumerate(add_n_values, start=1)]
    over_sampling = pd.DataFrame(oversample_data, columns=model_cols)
    interactive_graph(data=over_sampling, algo='Oversampling')

    # Under sampling -- can play around with start and stop
    start = false_train_size / 3.
    stop = false_train_size / 1.1
    remove_n_values = np.linspace(start, stop, num=50, dtype=int)
    undersample_data = [sample(model_data, i_run, remove_n=remove_n)
                        for i_run, remove_n in
                        enumerate(remove_n_values, start=1)]
    under_sampling = pd.DataFrame(undersample_data, columns=model_cols)
    interactive_graph(data=under_sampling, algo='Undersampling')

    # SMOTE sampling
    ratios = np.linspace(0.15, 1., num=50)
    smote_data = [sample(model_data, i_run, ratio=ratio)
                  for i_run, ratio in enumerate(ratios, start=1)]
    smote_sampling = pd.DataFrame(smote_data, columns=model_cols)

    interactive_graph(data=smote_sampling, algo='SMOTE')
