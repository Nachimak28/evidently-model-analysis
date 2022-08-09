import os
import pandas as pd
from evidently_model_analysis import EvidentlyModelAnalysis
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.storage.payload import Payload
import lightning as L
from sklearn import ensemble
import tempfile

class StaticPageViewer(L.LightningFlow):
    def __init__(self, page_path: str):
        super().__init__()
        self.serve_dir = page_path

    def configure_layout(self):
        return StaticWebFrontend(serve_dir=self.serve_dir)


class TempWorkComponent(L.LightningWork):
    def __init__(self, parallel=True) -> None:
        super().__init__(parallel=parallel)
        self.train_df = None
        self.test_df = None

    def run(self):
        self.train_df = Payload(pd.read_csv('ba_cancer_train_df_with_preds.csv'))
        self.test_df = Payload(pd.read_csv('ba_cancer_test_df_with_preds.csv'))

class LitApp(L.LightningFlow):
    def __init__(self, train_dataframe_path=None, test_dataframe_path=None, target_column_name=None, prediction_column_name=None, task_type='classification') -> None:
        super().__init__()
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.prediction_column_name = prediction_column_name
        self.task_type = task_type
        
        self.evidently_model_analysis = EvidentlyModelAnalysis(
                                                        train_dataframe_path=self.train_dataframe_path,
                                                        test_dataframe_path=self.test_dataframe_path,
                                                        target_column_name=self.target_column_name,
                                                        prediction_column_name=self.prediction_column_name,
                                                        task_type=self.task_type,
                                                        parallel=False)

        self.report_render = StaticPageViewer(self.evidently_data_analysis.report_parent_path)
        
        self.temp_component = TempWorkComponent(parallel=False)

    def run(self):
        self.temp_component.run()
        self.evidently_model_analysis.task_type = 'classification'
        self.evidently_model_analysis.target_column_name = 'target'
        self.evidently_model_analysis.prediction_column_name = 'prediction'
        self.evidently_model_analysis.run(train_df=self.temp_component.train_df, test_df=self.temp_component.test_df)
        print(self.evidently_model_analysis.report_path)


    def configure_layout(self):
        tab_1 = {'name': 'Model report', 'content': self.report_render}
        return tab_1

if __name__ == "__main__":
    # classification use case
    cancer_df = pd.read_csv('resources/bcancer.csv')
    total_rows = len(cancer_df)
    train_length = int(total_rows*0.75)

    train_df, test_df = cancer_df[:train_length], cancer_df[train_length:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df.to_csv('ba_cancer_train.csv', index=False)
    test_df.to_csv('ba_cancer_test.csv', index=False)

    # demo training here
    all_columns = list(train_df.columns)
    all_columns.remove('target')
    feature_columns = all_columns

    # model building
    model = ensemble.RandomForestClassifier(random_state=0)
    model.fit(train_df[feature_columns], train_df.target)

    train_predictions = model.predict(train_df[feature_columns])
    test_predictions = model.predict(test_df[feature_columns])

    train_df['prediction'] = train_predictions
    test_df['prediction'] = test_predictions

    # save preds
    train_df.to_csv('ba_cancer_train_df_with_preds.csv', index=False)
    test_df.to_csv('ba_cancer_test_df_with_preds.csv', index=False)

    # app = L.LightningApp(LitApp(
    #         train_dataframe_path='ba_cancer_train_df_with_preds.csv',
    #         test_dataframe_path='ba_cancer_test_df_with_preds.csv',
    #         target_column_name='target',
    #         prediction_column_name='prediction',
    #         task_type='classification'
    #     ))

    app = L.LightningApp(LitApp())

    # regression use case
    # similar case can be made for regresion

