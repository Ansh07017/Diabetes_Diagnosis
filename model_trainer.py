import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DiabetesModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        # Models
        self.lr_model = None
        self.dt_model = None
        
        # Metrics
        self.lr_metrics = {}
        self.dt_metrics = {}
        
        # Cross-validation scores
        self.cv_scores = {}
        
    def load_and_preprocess_data(self):
        """Load dataset and handle missing values"""
        self.df = pd.read_csv(self.dataset_path)
        
        # Handle missing values
        # For Age: fill with median
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        
        # For BMI: fill with median
        self.df['BMI'] = self.df['BMI'].fillna(self.df['BMI'].median())
        
        # For Glucose: fill with median
        self.df['Glucose'] = self.df['Glucose'].fillna(self.df['Glucose'].median())
        
        # For BloodPressure: fill with median
        self.df['BloodPressure'] = self.df['BloodPressure'].fillna(self.df['BloodPressure'].median())
        
        # Convert FamilyHistory to binary (Yes=1, No=0)
        self.df['FamilyHistory'] = self.df['FamilyHistory'].map({'Yes': 1, 'No': 0})
        
        # Remove any rows with all missing values
        self.df = self.df.dropna(how='all')
        
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X = self.df[['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory']]
        y = self.df['Diabetic']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self, use_grid_search=True):
        """Train Logistic Regression model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            self.lr_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
            self.lr_model.fit(self.X_train_scaled, self.y_train)
            best_params = {}
        
        # Cross-validation
        cv_scores = cross_val_score(self.lr_model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['logistic_regression'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Predictions
        y_pred = self.lr_model.predict(self.X_test_scaled)
        y_pred_proba = self.lr_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.lr_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'roc_curve': roc_curve(self.y_test, y_pred_proba),
            'auc': auc(*roc_curve(self.y_test, y_pred_proba)[:2]),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True, zero_division=0),
            'best_params': best_params,
            'cv_score': cv_scores.mean()
        }
        
        return self.lr_model, self.lr_metrics
    
    def train_decision_tree(self, use_grid_search=True):
        """Train Decision Tree model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.dt_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            self.dt_model = DecisionTreeClassifier(
                random_state=42, 
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5
            )
            self.dt_model.fit(self.X_train, self.y_train)
            best_params = {}
        
        # Cross-validation
        cv_scores = cross_val_score(self.dt_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['decision_tree'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Predictions
        y_pred = self.dt_model.predict(self.X_test)
        y_pred_proba = self.dt_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        self.dt_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'roc_curve': roc_curve(self.y_test, y_pred_proba),
            'auc': auc(*roc_curve(self.y_test, y_pred_proba)[:2]),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True, zero_division=0),
            'best_params': best_params,
            'cv_score': cv_scores.mean()
        }
        
        return self.dt_model, self.dt_metrics
    
    def predict_patient(self, age, bmi, glucose, blood_pressure, family_history):
        """Predict diabetes risk for a new patient using both models"""
        # Prepare input
        input_data = pd.DataFrame({
            'Age': [age],
            'BMI': [bmi],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'FamilyHistory': [1 if family_history == 'Yes' else 0]
        })
        
        # Scale for Logistic Regression
        input_scaled = self.scaler.transform(input_data)
        
        # Logistic Regression prediction
        lr_prediction = self.lr_model.predict(input_scaled)[0]
        lr_probability = self.lr_model.predict_proba(input_scaled)[0]
        
        # Decision Tree prediction
        dt_prediction = self.dt_model.predict(input_data)[0]
        dt_probability = self.dt_model.predict_proba(input_data)[0]
        
        return {
            'logistic_regression': {
                'prediction': int(lr_prediction),
                'probability_negative': lr_probability[0] * 100,
                'probability_positive': lr_probability[1] * 100
            },
            'decision_tree': {
                'prediction': int(dt_prediction),
                'probability_negative': dt_probability[0] * 100,
                'probability_positive': dt_probability[1] * 100
            }
        }
    
    def get_feature_importance(self):
        """Get feature importance from Decision Tree"""
        if self.dt_model:
            feature_names = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory']
            importances = self.dt_model.feature_importances_
            return dict(zip(feature_names, importances))
        return None
    
    def train_all_models(self, use_grid_search=True):
        """Complete training pipeline with optional hyperparameter tuning"""
        self.load_and_preprocess_data()
        self.split_data()
        self.train_logistic_regression(use_grid_search=use_grid_search)
        self.train_decision_tree(use_grid_search=use_grid_search)
        
        return {
            'lr_metrics': self.lr_metrics,
            'dt_metrics': self.dt_metrics,
            'feature_importance': self.get_feature_importance(),
            'cv_scores': self.cv_scores
        }
