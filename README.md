# F1 Race Outcome Predictor

A data-driven project to predict Formula 1 race results using historical data, weather information and recent driver/team form.
The core model is a Linear Regression trained on engineered features such as grid position, rolling performance metrics and weather variables, achieving a mean absolute error of roughly 2.2–2.5 race positions on recent seasons.

## Installatie-voorbeeld:

pip install -r requirements.txt
1. Data voorbereiden
1.1. Echte resultaten 2024 opslaan
Dit script slaat de echte race-uitslagen op van 2024 naar PROCESSED_DIR/2024_actual_results.csv.

bash

    python -m scripts.save_2024_results

Gebruik dit wanneer je de 2024-validatie opnieuw wilt genereren.

2. Model trainen (volledige pipeline)
run_training.py voert de hele pipeline uit:

features bouwen met FastF1 + Kaggle

Model trainen op trainingsseizoenen

Evaluatie op testseizoenen

Run:

bash

    python -m scripts.run_training

Let op:
De features worden als parquet opgeslagen op de locatie uit config.FEATURES_PATH.

3. Voorspellingen draaien
3.1. Heel seizoen voorspellen
predict_season.py gebruikt het getrainde model om een compleet seizoen te voorspellen en slaat de uitkomst op als CSV.

bash

    python -m scripts.predict_season

Interactieve input:

text
Enter race season (e.g. 2024):
– kies het seizoen.

Output:

CSV: PROCESSED_DIR/2024_predictions.csv

Console: top 3 voorspelde coureurs per race (round).

3.2. Één race voorspellen
predict_race.py voorspelt de uitslag van één Grand Prix.

bash

    python -m scripts.predict_race

Interactieve input:

text
Enter GP name (e.g., 'Monaco', 'Silverstone', 'Monaco Grand Prix')
Wordt gemapt via GP_NAME_MAPPING naar (event_name, round).

text
Enter season (e.g., 2024) or press Enter for 2024
Output (console):

Geordende lijst met voorspelde finish-volgorde:

positie, driver code, teamnaam, predictiescore.

5. Typische workflow
(Eénmalig per setup) Dependencies installeren.

(Optioneel, wanneer data verandert) save_2024_results en dan run_training uitvoeren.

Tijdens ontwikkeling:

predict_season gebruiken voor volledige seizoen-run.

predict_race gebruiken voor snelle racechecks en oral exam demo.
