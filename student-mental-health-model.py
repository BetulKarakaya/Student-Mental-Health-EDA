import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

class DropoutRiskPredictor:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.best_model = None
        
    def create_pipeline(self, features) -> None:
        self.preprocessor = ColumnTransformer(
            transformers=[('num', RobustScaler(), features)])
        
        self.model = lgb.LGBMRegressor(
            random_state=42, 
            n_jobs=-1, 
            verbosity=-1)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', self.model)
        ])
        
        self.param_grid = {
            'regressor__n_estimators': [1000, 1500],
            'regressor__learning_rate': [0.03, 0.05],
            'regressor__max_depth': [6, 8], 
            'regressor__num_leaves': [31, 63], 
            'regressor__reg_alpha': [1.0, 2.0],
            'regressor__reg_lambda': [1.0, 2.0]
        }

    def train(self) -> None:
        start_time = time.time()
        print("Training with constraints and custom features...")
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=self.param_grid,
            n_iter=10,           
            cv=3, 
            scoring='r2',
            verbose=1,
            n_jobs=-1,
            random_state=42)
        
        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV R2 Score: {search.best_score_:.4f}")
        print(f"Total Processing Time: {int(elapsed // 60)} dk {elapsed % 60:.2f} sn")

    def predict(self) -> None:
        if self.best_model is None:
            print("Önce modeli eğitmelisin!")
            return
            
        self.y_pred = self.best_model.predict(self.X_test)
        self.train_pred = self.best_model.predict(self.X_train)
   
    def report_accuracy(self) -> None:
        print("-" * 35)
        print(f"Train R2 Score: {r2_score(self.y_train, self.train_pred):.4f}")
        print(f"Test R2 Score:  {r2_score(self.y_test, self.y_pred):.4f}")
        print(f"MAE (Mean Error): {mean_absolute_error(self.y_test, self.y_pred):.4f}")
        print("-" * 35)

def read_data(file_path, **kwargs):
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, **kwargs)

def optimize_floats(df):
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def optimize_ints(df):
    ints = df.select_dtypes(include=['int64']).columns
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def select_important_features(df, features_list, target='dropout_risk'):
    cols_to_keep = [f for f in features_list if f in df.columns] + [target]
    return df[cols_to_keep].copy()

def add_custom_features(df):
    
    df.loc[:, 'stress_sleep_ratio'] = df['stress_level'] / (df['sleep_hours'] + 1)
    df.loc[:, 'fin_social_balance'] = df['financial_stress'] / (df['social_support'] + 1)
    df.loc[:, 'total_mental_load'] = df['depression_score'] + df['anxiety_score'] + df['stress_level']
    df.loc[:, 'exam_sleep_impact'] = df['exam_pressure'] * (10 - df['sleep_hours'])
    return df
    
def main():
    df = read_data(file_path = "student_mental_health_burnout_1M.csv")
    
    base_features = [
        'depression_score', 'stress_level', 'anxiety_score', 
        'financial_stress', 'social_support', 'sleep_hours', 
        'exam_pressure'
    ]
    target = 'dropout_risk'
    
    df_final = (df.pipe(optimize_floats)
                  .pipe(optimize_ints)
                  .pipe(select_important_features, features_list=base_features)
                  .pipe(add_custom_features))
    
    custom_features = ['stress_sleep_ratio', 'fin_social_balance', 'total_mental_load', 'exam_sleep_impact']
    all_features = base_features + custom_features
    
    X = df_final[all_features]
    y = df_final[target]
    
    predictor = DropoutRiskPredictor(X=X, y=y)
    predictor.create_pipeline(features = all_features)
    predictor.train()
    predictor.predict()
    predictor.report_accuracy()

if __name__ == "__main__":
    main()