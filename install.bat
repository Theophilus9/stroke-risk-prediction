@echo off
REM Create virtual environment
python -m venv ml_env

REM Activate the virtual environment
call ml_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install core packages
pip install pandas numpy scikit-learn matplotlib django

REM Install additional ML/AI packages
pip install seaborn
pip install jupyterlab
pip install notebook
pip install pandas
pip install joblib
pip install plotly
pip install shap lime
pip install flask
pip install numpy
pip install scikit-learn

echo All packages installed successfully.
pause
