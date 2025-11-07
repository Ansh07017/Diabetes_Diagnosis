# Diabetes Risk Prediction System

## Overview

This is a machine learning-based web application for predicting diabetes risk. The system uses a Streamlit frontend to provide an interactive medical-grade interface where users can input patient data and receive risk predictions. The application employs multiple classification algorithms (Logistic Regression and Decision Tree) to analyze health metrics including age, BMI, glucose levels, blood pressure, and family history. The system is designed for healthcare professionals and researchers to assess diabetes risk factors through a user-friendly interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Design Pattern**: Single-page application with medical-themed styling
- **Styling**: Custom CSS with predefined color scheme optimized for healthcare interfaces (blues, greens, professional palette)
- **Font System**: Inter and Source Sans Pro fonts for professional readability
- **Layout**: Wide layout with expandable sidebar for navigation and input controls
- **Component Structure**: Modular UI components with interactive widgets for data input and visualization

### Backend Architecture
- **Model Training Module**: Object-oriented `DiabetesModelTrainer` class that encapsulates all ML operations
- **Data Processing Pipeline**: 
  - CSV file loading via pandas
  - Missing value imputation using median strategy for numerical features
  - Binary encoding for categorical features (FamilyHistory: Yes/No → 1/0)
  - Feature scaling using StandardScaler for normalization
- **Machine Learning Models**:
  - Logistic Regression: Linear classifier for baseline prediction
  - Decision Tree Classifier: Non-linear classifier for complex pattern detection
  - Rationale: Dual model approach allows comparison and ensemble possibilities
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, confusion matrix, ROC curves, and AUC scores
- **Data Split Strategy**: Train-test split approach for model validation

### Data Storage
- **Primary Storage**: CSV file-based dataset storage
- **No Database**: Application uses file-based data persistence (suitable for prototype/research environments)
- **Data Processing**: In-memory pandas DataFrames for data manipulation
- **Feature Set**: Age, BMI, Glucose, BloodPressure, FamilyHistory as predictive features
- **Pros**: Simple deployment, no database setup required, portable data files
- **Cons**: Not suitable for high-volume or concurrent user scenarios, limited scalability

### Visualization Architecture
- **Plotting Libraries**: Matplotlib and Seaborn for statistical visualizations
- **Chart Types**: Supports medical data visualization including distributions, correlations, confusion matrices, and ROC curves
- **Integration**: Streamlit's native plotting support for seamless chart rendering

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework and UI components
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing and array operations
- **matplotlib**: Static plotting and visualization
- **seaborn**: Statistical data visualization with enhanced aesthetics
- **scikit-learn**: Machine learning algorithms, preprocessing, and evaluation metrics
  - `LogisticRegression`: Linear classification model
  - `DecisionTreeClassifier`: Tree-based classification model
  - `StandardScaler`: Feature normalization
  - `train_test_split`: Data splitting utility
  - Various metrics modules for model evaluation

### External Services
- **Google Fonts API**: Custom font loading (Inter and Source Sans Pro families)
- No external APIs for data fetching or storage
- No authentication services (application assumes trusted environment)
- No cloud storage integrations

### Data Requirements
- Expects CSV dataset with columns: Age, BMI, Glucose, BloodPressure, FamilyHistory
- Dataset path must be provided to DiabetesModelTrainer constructor
- No external database connections required