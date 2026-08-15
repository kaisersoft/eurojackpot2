
# Eurojackpot Predictor 2.1

## Enthaltene historische Daten

`data/eurojackpot_history.csv` enthält **981 validierte Ziehungen vom 2012-03-23 bis 2026-08-14**.

CSV-Schema:

`Datum;zahl1;zahl2;zahl3;zahl4;zahl5;euro1;euro2`

Die aktuelle Datenkopie stammt aus einem öffentlich verfügbaren historischen Eurojackpot-Datensatz und wurde für diese App normalisiert und validiert. WestLotto stellt selbst historische Gewinnzahlen seit dem Start der Lotterie sowie CSV/Excel-Downloads bereit. 

## V2.1

- Ensemble aus 8 Modellen
- Monte-Carlo-Kandidaten
- Zahlen-/Paar-/Triple-Statistik
- Recent und Exponential Decay
- Overdue/Cold als Gegenmodelle
- Strukturmerkmale
- Walk-forward Backtesting
- Prediction History
- eingebettete historische CSV
- automatischer GitHub-Action-Datenupdate zweimal pro Woche
- Datenvalidierung und Duplikatbereinigung

## Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit

Repository-Inhalt hochladen und `app.py` als Main File in Streamlit Community Cloud auswählen.

Die GitHub Action unter `.github/workflows/update_data.yml` aktualisiert die CSV automatisch. 

## Datenquelle / wissenschaftliche Vorsicht

WestLotto weist darauf hin, dass die Wahrscheinlichkeit einer auswählbaren Zahl nicht davon abhängt, wann sie zuletzt gezogen wurde. Die Scores der App sind deshalb **keine echten Gewinnwahrscheinlichkeiten**, sondern statistische Rankingwerte.

Die Spielformel änderte sich am 10.10.2014 (Eurozahlen 2 aus 10 statt 2 aus 8) und am 25.03.2022 (Eurozahlen 11 und 12 kamen hinzu). V2.1 berücksichtigt diese Regimeänderungen bei der Interpretation.
