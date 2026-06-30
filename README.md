# Intelligente Müllsortierung
 
**Entwicklung eines intelligenten Müllsortierungssystems mittels Bildklassifikation**
 
Dieses Projekt wurde im Rahmen des Moduls **Applied Machine Learning** entwickelt.
 
Es implementiert ein Bildklassifikationsmodell auf Basis von **MobileNetV2 (Transfer Learning)** zur intelligenten Erkennung von fünf Müllklassen.
 
Folgende Klassen werden unterschieden:
 
- Glas
- Metall
- Papier
- Plastik
- Restmüll
 
Zusätzlich werden verschiedene **Data-Augmentation-Stufen** sowie unterschiedliche **Trainingsdatensätze** miteinander verglichen.
 
 
## Inhalt
 
- Voraussetzungen
- Projektstruktur
- Datensatz
- Installation
- Konfiguration
- Training
- Bildklassifikation
- Erwartete Ausgaben
- Troubleshooting
 
 
## Voraussetzungen
 
- Windows (PowerShell)
- Python 3.11+
- TensorFlow
- Internetzugang zum Installieren der Python-Pakete
 
 
## Projektstruktur
 
```text
AML-Waste-Classification/
 
├── train_model.py
├── requirements.txt
├── README.md
├── muell_model_projekt.keras
├── test/
│   ├── glas/
│   ├── metall/
│   ├── papier/
│   ├── plastik/
│   └── restmuell/
```
 
## Datensatz
 
Dieses Projekt verwendet einen öffentlichen Datensatz von **Kaggle**, welcher durch **eigene Aufnahmen** ergänzt wurde.
 
Die Bilder werden in folgende Klassen eingeteilt:
 
- Glas
- Metall
- Papier
- Plastik
- Restmüll
 
Eigenschaften:
 
- RGB
- JPG
- Bildgröße: 224 × 224 Pixel
 
 
## Installation
 
### 1) Virtuelle Umgebung erstellen
 
```powershell
python -m venv .venv
```
 
Aktivieren:
 
```powershell
.\.venv\Scripts\activate
```
 
### 2) Pakete installieren
 
```powershell
pip install -r requirements.txt
```
 
### 3) Installation überprüfen
 
```powershell
python -c "import tensorflow, numpy, matplotlib; print('OK')"
```
 
 
## Konfiguration
 
Die wichtigsten Trainingsparameter befinden sich direkt im Python-Code.
 
Verwendete Parameter:
 
- MobileNetV2
- Transfer Learning
- Batch Size = 16
- Learning Rate = 0.0001
- 25 Epochen
- Dropout = 0.3
- Data Augmentation
- Class Weight für Restmüll
 
 
## Training
 
Training starten:
 
```powershell
python train_model.py
```
 
Während des Trainings werden automatisch Accuracy und Loss berechnet.
 
 
## Bildklassifikation
 
Nach Abschluss des Trainings werden alle Bilder im Testordner automatisch klassifiziert.
 
Ausgegeben werden:
 
- Bildname
- vorhergesagte Klasse
- Vorhersagewahrscheinlichkeit
 
Beispiel:
 
```text
Bilddatei: test1.jpg
Klassifikation: Glas
Sicherheit: 98.21 %
 
Bilddatei: test2.jpg
Klassifikation: Plastik
Sicherheit: 99.40 %
```
 
## Erwartete Ausgaben
 
Nach erfolgreichem Training werden erzeugt:
 
- trainiertes Modell (`muell_model_projekt.keras`)
- Accuracy-Verlauf
- Loss-Verlauf
- Confusion Matrix
- Klassifikation der Testbilder
 
 
## Troubleshooting
 
### ModuleNotFoundError
 
Virtuelle Umgebung aktivieren:
 
```powershell
.\.venv\Scripts\activate
```
 
Pakete erneut installieren:
 
```powershell
pip install -r requirements.txt
```
 
 
### TensorFlow wird nicht gefunden
 
Prüfen:
 
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```
 
 
 
### Trainingsdaten werden nicht gefunden
 
Sicherstellen, dass sich die Trainings- und Testbilder in den entsprechenden Ordnern befinden und die Ordnerstruktur unverändert ist.


### Hinweis

Der vollständige Datensatz ist aufgrund der Dateigröße nicht direkt im GitHub-Repository enthalten und wird separat bereitgestellt. 
 