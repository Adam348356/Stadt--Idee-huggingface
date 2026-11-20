# StadtIdee · Hugging Face Edition

Eine Streamlit-App für Stadtideen: Ein Foto eines öffentlichen Orts hochladen oder aufnehmen, Ergänzungen einzeichnen, einzeln klassifizieren und den Vorschlag veröffentlichen.

## Lokal starten (Ubuntu)

```bash
sudo apt install python3-venv
cd ~/Downloads/StadtIdee-HuggingFace
bash start_app.sh
```

Das Skript erstellt bei Bedarf `.venv`, installiert die Abhängigkeiten und startet Streamlit. Danach die im Terminal angezeigte Adresse öffnen, normalerweise `http://localhost:8501`.

Beim ersten Klassifikationslauf lädt die App `kmewhort/beit-sketch-classifier` von Hugging Face herunter. Dafür braucht die Laufzeit Internetzugriff; danach wird das Modell lokal zwischengespeichert. Das Modell ordnet jede Zeichnung einzeln einer von 345 QuickDraw-Skizzenkategorien zu. Kategorien, die nicht übersetzt sind, erscheinen in der Originalbezeichnung des Modells.

## Fotos aufnehmen

Unter **Eigene Projekte** kannst du **Foto aufnehmen** auswählen. Erlaube den Kamerazugriff im Browser. Auf `localhost` funktioniert das direkt; für eine veröffentlichte Website benötigt die Kamera HTTPS.

## Repository-Struktur

```text
StadtIdee-HuggingFace/
├── app.py
├── classifier.py
├── image_processing.py
├── project_store.py
├── ui_theme.py
├── requirements.txt
├── runtime.txt
├── start_app.sh
├── .gitignore
└── README.md
```

Das Klassifikationsmodell wird zur Laufzeit von Hugging Face geladen; ein Modellgewichtsordner muss nicht ins GitHub-Repository. Die SQLite-Projektdatenbank wird beim Start automatisch erzeugt.

## Zu GitHub hochladen

Erstelle auf GitHub ein leeres Repository. Führe dann im Ubuntu-Terminal aus (Platzhalter in der Repository-Adresse ersetzen):

```bash
cd ~/Downloads/StadtIdee-HuggingFace
git init
git add .
git commit -m "StadtIdee mit Hugging Face Sketch-Klassifikation"
git branch -M main
git remote add origin https://github.com/DEIN-NAME/DEIN-REPOSITORY.git
git push -u origin main
```

Die Dateien müssen direkt im Stammverzeichnis des GitHub-Repositories liegen. Verbinde es anschließend mit Streamlit Community Cloud und wähle `app.py` als Hauptdatei. `runtime.txt` gibt Python 3.11 für das Hosting vor.
