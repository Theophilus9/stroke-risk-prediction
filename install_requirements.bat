@echo off

REM Upgrade pip globally
python -m pip install --upgrade pip

REM Install core packages globally
pip install pandas numpy scikit-learn matplotlib

REM Install additional ML/AI packages globally
pip install seaborn joblib flask

echo All packages installed successfully.
pause
