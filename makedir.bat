@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo 🚀 Setting up Local Project Structure for Regime ML
echo ===============================================

REM === Root folders ===
mkdir data
mkdir data\raw
mkdir data\cleaned
mkdir data\features
mkdir data\processed
mkdir data\tasks

mkdir notebooks

mkdir src
mkdir src\data
mkdir src\models
mkdir src\utils

mkdir tests
mkdir configs
mkdir docs

REM === Placeholder files ===
echo > data\raw\sp500_raw.csv
echo > data\raw\vix_raw.csv
echo > data\processed\regime_segments.pkl

echo > src\data\__init__.py
echo > src\models\__init__.py
echo > src\utils\__init__.py
echo > src\__init__.py

echo # Data acquisition and preprocessing scripts > src\data\data_acquisition.py
echo # Data cleaning pipeline > src\data\data_cleaning.py
echo # Feature engineering pipeline > src\data\feature_engineering.py
echo # Task dataset > src\data\task_dataset.py

echo # Hidden Markov Model implementation > src\models\hmm_regime.py
echo # Portfolio network (MLP) > src\models\portfolio_net.py

echo # Logging and utilities > src\utils\experiment_logger.py
echo # Plotting functions > src\utils\plotting.py
echo # Validation functions > src\utils\validations.py

echo # Unit tests for data pipeline > tests\test_data_pipeline.py
echo # Unit tests for feature engineering > tests\test_features.py
echo # Unit tests for task dataset > tests\test_task_dataset.py
echo # Unit tests for models > tests\test_models.py

echo experiment_name: "default_experiment" > configs\experiment_template.yaml
echo > configs\hyperparams_tracker.xlsx

echo # Documentation files > docs\01_data_pipeline.md
echo # Documentation files > docs\02_feature_engineering.md
echo # Documentation files > docs\03_hmm_regime_detection.md
echo # Documentation files > docs\04_meta_learning_tasks.md

echo # README > README.md
(
echo numpy
echo pandas
echo matplotlib
echo seaborn
echo yfinance
echo hmmlearn
echo torch
echo scikit-learn
) > requirements.txt

echo ✅ Project structure created successfully!

REM === Optional Git commit & push ===
choice /M "Do you want to commit and push these changes to GitHub now?"
if errorlevel 1 (
    git add .
    git commit -m "Setup: initial project directory structure for Regime ML"
    git push
    echo ✅ Changes pushed to GitHub!
)

pause
