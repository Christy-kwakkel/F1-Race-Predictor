# Personal-Project

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

venv\Scripts\Activate.ps1

python -m scripts.predict_race
python -m scripts.run_training
python -m scripts.save_2024_results
python -m scripts.predict_season

