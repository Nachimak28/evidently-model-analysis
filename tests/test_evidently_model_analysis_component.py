r"""
To test a lightning component:

1. Init the component.
2. call .run()
"""

import os
import random
import numpy as np
import pandas as pd
from lightning.app.storage.payload import Payload
from evidently_model_analysis.component import EvidentlyModelAnalysis


# test when args passed during init for classification
def test_component_init_classification_example():
    ema = EvidentlyModelAnalysis(
        train_dataframe_path='../resources/ba_cancer_train_df_with_preds.csv',
        test_dataframe_path='../resources/ba_cancer_test_df_with_preds.csv',
        target_column_name='target',
        prediction_column_name='prediction',
        task_type='classification',
        parallel=False
    )
    ema.build_dashboard()
    assert ema.report_path != None and 'index.html' in ema.report_path

# test when args passed during init for regression
def test_component_init_regression_example():
    ema = EvidentlyModelAnalysis(
        train_dataframe_path='../resources/ca_housing_train_with_preds.csv',
        test_dataframe_path='../resources/ca_housing_test_with_preds.csv',
        target_column_name='MedHouseVal',
        prediction_column_name='prediction',
        task_type='regression',
        parallel=False
    )
    ema.build_dashboard()
    assert ema.report_path != None and 'index.html' in ema.report_path

# test if report is built as path specified
def test_report_path():
    report_parent_path = './temp_report_parent'
    ema = EvidentlyModelAnalysis(
        train_dataframe_path='../resources/ca_housing_train_with_preds.csv',
        test_dataframe_path='../resources/ca_housing_test_with_preds.csv',
        target_column_name='MedHouseVal',
        prediction_column_name='prediction',
        task_type='regression',
        parallel=False,
        report_parent_path=report_parent_path
    )
    ema.build_dashboard()
    parent_path, report_filename = os.path.split(ema.report_path)
    assert parent_path == report_parent_path
    assert report_filename == 'index.html'
    # cleanup after asseertion
    os.remove(ema.report_path)
    os.removedirs(ema.report_parent_path)


# test when args passed during run
def test_component_args_during_run():
    # loading dataframes
    train_df = pd.read_csv('../resources/ba_cancer_train_df_with_preds.csv')
    test_df = pd.read_csv('../resources/ba_cancer_test_df_with_preds.csv')

    target_column_name = 'target'
    task_type = 'classification'
    prediction_column_name = 'prediction'

    ema = EvidentlyModelAnalysis()
    ema.target_column_name = target_column_name
    ema.task_type = task_type
    ema.prediction_column_name = prediction_column_name
    
    ema.build_dashboard(train_df=Payload(train_df), test_df=Payload(test_df))
    assert ema.report_path != None and 'index.html' in ema.report_path

# test when args are not valid payload objects
def test_idenfity_invalid_payload_inputs():
    # passing string and int to run method
    target_column_name = 'target'
    task_type = 'classification'
    prediction_column_name = 'prediction'

    ema = EvidentlyModelAnalysis()
    ema.target_column_name = target_column_name
    ema.task_type = task_type
    ema.prediction_column_name = prediction_column_name
    
    train_df = 'ABCD'
    test_df = 1234
    try:
        ema.build_dashboard(train_df=train_df, test_df=test_df)
    except Exception as e:
        assert isinstance(e, TypeError)
    

# test when args are not valid pandas dataframes
def test_idenfity_invalid_dataframe_inputs():
    # passing string and int to run method
    target_column_name = 'target'
    task_type = 'classification'
    prediction_column_name = 'prediction'

    ema = EvidentlyModelAnalysis()
    ema.target_column_name = target_column_name
    ema.task_type = task_type
    ema.prediction_column_name = prediction_column_name
    
    train_df = Payload('ABCD')
    test_df = Payload(1234)
    try:
        ema.build_dashboard(train_df=train_df, test_df=test_df)
    except Exception as e:
        assert isinstance(e, TypeError)


# test to check the compoment behaviour if NaN/null values are present in input dataframes
def test_component_behavior_nan():
    # loading a valid dataframe with no nans
    train_df = pd.read_csv('../resources/ba_cancer_train_with_nans.csv')
    test_df = pd.read_csv('../resources/ba_cancer_test_df_with_preds.csv')
    
    # inserting NaNs at random places in the dataframe
    ix = [(row, col) for row in range(train_df.shape[0]) for col in range(train_df.shape[1])]
    for row, col in random.sample(ix, int(round(.1*len(ix)))):
        train_df.iat[row, col] = np.nan
    # testing the component

    target_column_name = 'target'
    task_type = 'classification'
    prediction_column_name = 'prediction'

    ema = EvidentlyModelAnalysis()
    ema.target_column_name = target_column_name
    ema.task_type = task_type
    ema.prediction_column_name = prediction_column_name

    try:
        ema.build_dashboard(train_df=Payload(train_df), test_df=Payload(test_df))
    except Exception as e:
        assert isinstance(e, AssertionError)