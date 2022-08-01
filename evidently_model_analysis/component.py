import os
import lightning as L
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import RegressionPerformanceTab,  ClassificationPerformanceTab
import tempfile


class EvidentlyModelAnalysis(L.LightningWork):
    def __init__(self, train_dataframe_path=None, test_dataframe_path=None, target_column_name=None, prediction_column_name=None, task_type='classification') -> None:
        super().__init__()
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column = target_column_name
        self.task_type = task_type
        self.prediction_column_name = prediction_column_name
        self.report_path = None
        tmp_dir = tempfile.mkdtemp()
        self.report_parent_path = os.path.join(tmp_dir, 'model_performance')
        os.makedirs(self.report_parent_path, exist_ok=True)

        self.supported_task_types = ['classification', 'regression']

        if self.task_type not in self.supported_task_types:
            raise Exception(f'task_type must be {",".join(self.supported_task_types)}')


    def run(self):
        col_map = ColumnMapping()
        col_map.target = self.target_column
        col_map.prediction = self.prediction_column_name
        train_df = pd.read_csv(self.train_dataframe_path)
        test_df = pd.read_csv(self.test_dataframe_path)
        train_df.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        
        tabs = []
        if self.task_type == 'classification':
            tabs.append(ClassificationPerformanceTab(verbose_level=1))
        else:
            tabs.append(RegressionPerformanceTab(verbose_level=1))
        model_performance_dashboard = Dashboard(tabs=tabs)
        model_performance_dashboard.calculate(train_df, test_df, column_mapping=col_map)
        self.report_path = os.path.join(self.report_parent_path, 'index.html')
        model_performance_dashboard.save(self.report_path)