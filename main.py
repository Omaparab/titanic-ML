import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
from werkzeug.utils import secure_filename
import pickle

# --- FLASK SETUP ---
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- DATASET STORAGE CONFIGURATION ---
DATASET_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Create datasets folder if it doesn't exist
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# --- GLOBAL DATA STORAGE ---
df_original = None
trained_model = None
feature_columns = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_titanic_data():
    """Load Titanic dataset from local file if it exists, otherwise from URL"""
    global df_original
    
    # Try to load from local titanic.csv first (which includes predictions)
    if os.path.exists('titanic.csv'):
        df_original = pd.read_csv('titanic.csv')
    else:
        # Fallback to URL if local file doesn't exist
        url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
        df_original = pd.read_csv(url)
    
    return df_original

def prepare_baseline_data(df):
    """Prepare baseline features without feature generation"""
    df_baseline = df.copy()
    
    # Data Cleaning
    df_baseline['Age'] = df_baseline['Age'].fillna(df_baseline['Age'].median())
    df_baseline['Fare'] = df_baseline['Fare'].fillna(df_baseline['Fare'].median())
    
    # Feature Preparation
    df_baseline = pd.get_dummies(df_baseline, columns=['Sex'], drop_first=True)
    
    # Define features
    baseline_features = ['Pclass', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare', 'Sex_male']
    X = df_baseline[baseline_features]
    y = df_baseline['Survived']
    
    return X, y, df_baseline

def prepare_featured_data(df):
    """Prepare data with feature generation"""
    df_featured = df.copy()
    
    # Feature Generation
    # 1. FamilySize
    df_featured['FamilySize'] = df_featured['Siblings/Spouses Aboard'] + df_featured['Parents/Children Aboard'] + 1
    
    # 2. IsAlone
    df_featured['IsAlone'] = 0
    df_featured.loc[df_featured['FamilySize'] == 1, 'IsAlone'] = 1
    
    # 3. Title from Name
    df_featured['Title'] = df_featured['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_featured['Title'] = df_featured['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                        'Jonkheer', 'Dona'], 'Other')
    df_featured['Title'] = df_featured['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_featured['Title'] = df_featured['Title'].replace('Mme', 'Mrs')
    
    # Data Cleaning
    df_featured['Age'] = df_featured['Age'].fillna(df_featured['Age'].median())
    df_featured['Fare'] = df_featured['Fare'].fillna(df_featured['Fare'].median())
    
    # Convert categorical columns
    df_featured = pd.get_dummies(df_featured, columns=['Sex', 'Title'], drop_first=True)
    
    # Define features
    y = df_featured['Survived']
    X = df_featured.drop(columns=['Survived', 'Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'])
    
    return X, y, df_featured

def get_feature_explanations():
    """Returns explanations for generated features"""
    return {
        'FamilySize': {
            'description': 'Total family members aboard',
            'formula': 'Siblings/Spouses + Parents/Children + 1 (self)',
            'rationale': 'Larger families may have had different survival rates due to group dynamics'
        },
        'IsAlone': {
            'description': 'Binary indicator for solo travelers',
            'formula': '1 if FamilySize = 1, else 0',
            'rationale': 'Solo passengers might have had different survival chances than those with family'
        },
        'Title_Master': {
            'description': 'Extracted from passenger name',
            'formula': 'One-hot encoded title "Master" (young boys)',
            'rationale': 'Social titles reflect age, gender, and social status affecting survival priority'
        },
        'Title_Miss': {
            'description': 'Extracted from passenger name',
            'formula': 'One-hot encoded title "Miss" (unmarried women)',
            'rationale': 'Social titles reflect age, gender, and social status affecting survival priority'
        },
        'Title_Mr': {
            'description': 'Extracted from passenger name',
            'formula': 'One-hot encoded title "Mr" (adult men)',
            'rationale': 'Social titles reflect age, gender, and social status affecting survival priority'
        },
        'Title_Mrs': {
            'description': 'Extracted from passenger name',
            'formula': 'One-hot encoded title "Mrs" (married women)',
            'rationale': 'Social titles reflect age, gender, and social status affecting survival priority'
        },
        'Title_Other': {
            'description': 'Extracted from passenger name',
            'formula': 'Rare titles grouped (Dr, Rev, military ranks, nobility)',
            'rationale': 'Special titles may indicate privileged status affecting rescue priority'
        }
    }

def run_ensemble_ml(use_feature_generation=True, n_estimators=100, max_depth=None, min_samples_leaf=1):
    """
    Performs Feature Generation (optional), trains Random Forest, and returns metrics.
    """
    global trained_model, feature_columns
    
    df = load_titanic_data()
    
    # Prepare data based on feature generation option
    if use_feature_generation:
        X, y, df_processed = prepare_featured_data(df)
    else:
        X, y, df_processed = prepare_baseline_data(df)
    
    # Store feature columns for prediction
    feature_columns = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    trained_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth else None,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    trained_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = trained_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'accuracy': f"{accuracy:.4f}",
        'precision': f"{precision:.4f}",
        'recall': f"{recall:.4f}",
        'f1': f"{f1:.4f}",
        'confusion_matrix': cm.tolist(),
        'feature_names': feature_columns,
        'n_features': len(feature_columns)
    }
    
    return results, df_processed

def extract_title(name):
    """Extract title from passenger name"""
    import re
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        # Map rare titles
        if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
            return 'Other'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Mme':
            return 'Mrs'
        else:
            return title
    return 'Mr'

def prepare_input_for_prediction(data, use_feature_generation=True):
    """Prepare input data for model prediction"""
    # Create a temporary dataframe
    input_df = pd.DataFrame([{
        'Pclass': data['pclass'],
        'Name': data['name'],
        'Sex': data['sex'],
        'Age': data['age'],
        'Siblings/Spouses Aboard': data['sibsp'],
        'Parents/Children Aboard': data['parch'],
        'Fare': data['fare']
    }])
    
    if use_feature_generation:
        # Apply feature generation
        input_df['FamilySize'] = input_df['Siblings/Spouses Aboard'] + input_df['Parents/Children Aboard'] + 1
        input_df['IsAlone'] = (input_df['FamilySize'] == 1).astype(int)
        
        # Extract title
        input_df['Title'] = input_df['Name'].apply(extract_title)
        
        # One-hot encode Sex and Title
        input_df = pd.get_dummies(input_df, columns=['Sex', 'Title'], drop_first=True)
        
        # Drop Name and family columns
        input_df = input_df.drop(columns=['Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'], errors='ignore')
    else:
        # Baseline preparation
        input_df = pd.get_dummies(input_df, columns=['Sex'], drop_first=True)
        input_df = input_df.drop(columns=['Name'], errors='ignore')
    
    # Ensure all required columns exist (fill missing with 0)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Select only required columns in correct order
    input_df = input_df[feature_columns]
    
    return input_df

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """Redirect to dashboard as the default landing page."""
    from flask import redirect
    return redirect('/dashboard')

@app.route('/experiment', methods=['GET'])
def experiment():
    """Renders the experiment page."""
    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Renders the dashboard page."""
    return render_template('dashboard.html')

@app.route('/insert', methods=['GET'])
def insert():
    """Renders the data insertion page."""
    return render_template('insert.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handles dataset file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(DATASET_FOLDER, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                return jsonify({
                    'success': False, 
                    'message': f'Dataset "{filename}" already exists in the datasets folder'
                })
            
            # Save the file
            file.save(filepath)
            
            return jsonify({
                'success': True, 
                'message': f'Dataset "{filename}" uploaded successfully!'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Invalid file type. Only CSV and XLSX files are allowed'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    """Returns list of datasets in the datasets folder"""
    try:
        files = [f for f in os.listdir(DATASET_FOLDER) 
                if os.path.isfile(os.path.join(DATASET_FOLDER, f)) 
                and allowed_file(f)]
        
        return jsonify({'success': True, 'datasets': files})
    except Exception as e:
        return jsonify({'success': False, 'datasets': [], 'error': str(e)})

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    """Deletes a dataset from the datasets folder"""
    try:
        data = request.get_json()
        filename = secure_filename(data.get('filename', ''))
        
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        
        filepath = os.path.join(DATASET_FOLDER, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'Dataset "{filename}" deleted successfully'})
        else:
            return jsonify({'success': False, 'message': f'Dataset "{filename}" not found'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_data', methods=['GET'])
def get_data():
    """Returns dataset information for dashboard."""
    try:
        df = load_titanic_data()
        
        # Baseline data
        X_base, y_base, df_base = prepare_baseline_data(df)
        
        # Featured data
        X_feat, y_feat, df_feat = prepare_featured_data(df)
        
        # Original data - all and sample
        original_all = df.to_dict('records')
        original_sample = df.head(10).to_dict('records')
        
        # Baseline features info
        baseline_all = df_base[X_base.columns.tolist() + ['Survived']].to_dict('records')
        baseline_sample = df_base[X_base.columns.tolist() + ['Survived']].head(10).to_dict('records')
        
        # Featured data info
        featured_all = df_feat[X_feat.columns.tolist() + ['Survived']].to_dict('records')
        featured_sample = df_feat[X_feat.columns.tolist() + ['Survived']].head(10).to_dict('records')
        
        # Identify new features (those not in baseline)
        baseline_features = set(X_base.columns.tolist())
        featured_features = set(X_feat.columns.tolist())
        new_features = list(featured_features - baseline_features)
        
        return jsonify({
            'success': True,
            'original': {
                'sample': original_sample,
                'all_data': original_all,
                'columns': df.columns.tolist(),
                'n_rows': len(df)
            },
            'baseline': {
                'features': X_base.columns.tolist(),
                'sample': baseline_sample,
                'all_data': baseline_all,
                'n_features': len(X_base.columns)
            },
            'featured': {
                'features': X_feat.columns.tolist(),
                'sample': featured_sample,
                'all_data': featured_all,
                'n_features': len(X_feat.columns),
                'new_features': new_features
            },
            'feature_explanations': get_feature_explanations()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/run_ml', methods=['POST'])
def run_ml_task():
    """Handles the form submission, runs the ML, and returns results."""
    try:
        # Extract parameters from the form
        use_feature_generation = request.form.get('use_features') == 'true'
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        max_depth = int(max_depth) if max_depth and max_depth != 'None' else None
        min_samples_leaf = int(request.form.get('min_samples_leaf', 1))
        
        print(f"Running ML - Features: {use_feature_generation}, n_estimators: {n_estimators}")
        
        # Execute the core machine learning logic
        results, df_processed = run_ensemble_ml(
            use_feature_generation=use_feature_generation,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        
        # Return results as JSON
        return jsonify({
            'success': True,
            'results': results,
            'model_type': 'Random Forest with Feature Generation' if use_feature_generation else 'Random Forest (Baseline)'
        })
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_survival', methods=['POST'])
def predict_survival():
    """Predicts survival outcome for a given passenger using trained model and saves to dataset"""
    try:
        if trained_model is None or feature_columns is None:
            return jsonify({
                'success': False, 
                'error': 'Model not trained yet. Please run an experiment first.'
            })
        
        data = request.get_json()
        
        # Validate input
        required_fields = ['name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            })
        
        # Determine if feature generation was used based on feature_columns
        use_feature_generation = 'Title_Mr' in feature_columns or 'FamilySize' in feature_columns
        
        # Prepare input
        input_df = prepare_input_for_prediction(data, use_feature_generation)
        
        # Make prediction
        prediction = trained_model.predict(input_df)[0]
        probability = trained_model.predict_proba(input_df)[0][prediction]
        
        # Save to dataset
        try:
            # Load the current Titanic dataset
            df = load_titanic_data()
            
            # Create new record with standard columns
            new_record = {
                'Pclass': data['pclass'],
                'Name': data['name'],
                'Sex': data['sex'],
                'Age': data['age'],
                'Siblings/Spouses Aboard': data['sibsp'],
                'Parents/Children Aboard': data['parch'],
                'Fare': data['fare'],
                'Survived': int(prediction)
            }
            
            # Add the new record to the dataframe
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            
            # Save to both locations
            # 1. Save to datasets folder with predictions
            dataset_path = os.path.join(DATASET_FOLDER, 'titanic_with_predictions.csv')
            df.to_csv(dataset_path, index=False)
            print(f"Record saved to {dataset_path}")
            
            # 2. Save to root directory as titanic.csv (updating the original)
            root_dataset_path = 'titanic.csv'
            df.to_csv(root_dataset_path, index=False)
            print(f"Record saved to {root_dataset_path}")
        
        except Exception as save_error:
            print(f"Warning: Could not save record to file: {save_error}")
            # Continue with response even if save fails
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability)
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)