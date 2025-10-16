# Loan-Approval-ML-Kaggle-Dataset
Loan Approval ML Kaggle Dataset

# About Dataset
This dataset simulates loan applications and approval outcomes for 2,000 individuals. It contains demographic, financial, and employment-related attributes that can be used to predict whether a loan application will be approved or rejected. It is ideal for practicing classification problems, credit risk modeling, and feature engineering for financial datasets.

# Pre-requisite
pip install lightgbm
pip install catboost

# Library Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
