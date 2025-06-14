 Ideal Workflow: Anime-Film → Charakter-Dataset für SD-Training

 Übersicht


    A Anime-Film Video  --> B Frame Extraction 
    B --> C Frame Deduplication 
    C --> D Charakter-Klassifizierung 
    D --> E Upscaling + Quality Check 
    E --> F Cropping + Face Detection
    F --> G Image Annotation / Tagging
    G --> H Final Dataset Images + Caption

1.  Frame Extraction

Ziel: Einzelbilder Frames aus dem Anime-Film extrahieren
Tools: `ffmpeg`

Beispielbefehl:

ffmpeg -i anime_film.mp4 -vf fps=1 output/frame_%04d.png


 `fps=1`: 1 Frame pro Sekunde anpassen je nach Dynamik
 Qualität beachten: Verwende eine BluRay/1080p-Quelle



2.  Frame Deduplication

Ziel: Doppelte oder fast identische Frames entfernen
Tools: FiftyOne, `img-dupes`, `imgsim`, eigene Hash-Vergleiche

 SSIM oder Perceptual Hash phash verwenden
 Duplikate automatisch markieren und löschen

 Tipp: Lieber aggressiv aussortieren – Qualität > Quantität



3.  Charakter-Klassifizierung

Ziel: Bilder nach Charakteren aufteilen – Animes enthalten meist viele Charaktere

Methode:

 Manuelle Klassifikation über GUI z. B. FiftyOne mit Tagging
 Automatisierte Klassifikation durch Pre-LoRA Detection z. B. eigene CLIP-basierte Modelle oder Classifier
 Optional: Embedding Clustering + Human Review z. B. mit `CLIP+UMAP`

 Wichtig: Ohne korrekte Trennung der Charaktere leidet das Modell durch Stilvermischung



4.  Upscaling & Quality Check

Ziel: Bilder visuell verbessern
Tools: `Real-ESRGAN`, `Topaz`, eigene ESRGAN-Modelle

 Upscale auf 2x oder 4x falls nötig
 Schlechte, verrauschte oder zu dunkle Frames entfernen



5.  Cropping & Face Detection

Ziel: Fokus auf Gesicht, Oberkörper oder typische Charakter-Darstellung
Tools: `YOLOv5`, `mediapipe`, `animeface`, `Sefid`, eigenes Cropping-Skript

 Optional: Keyframe-Sets mit unterschiedlichen Zooms/Angles



6.  Image Annotation / Tagging

Ziel: Caption-Dateien erstellen für das Training
Tools:

 Automatisch: `WD14-Tagger`, `DeepDanbooru`
 Manuell prüfen & nachbessern
 Einheitliche Benennung: `charname_outfit_emotion_background.png`

Beispiel-Caption Textdatei pro Bild:


1girl, solo, {charname}, long_hair, school_uniform, smile, indoors, anime_style




7.  Finales Dataset

Struktur:

/dataset/
  /images/
    charname_0001.png
    ...
  /captions/
    charname_0001.txt
    ...

 30–100 Bilder für LoRA reichen oft aus bei Stiltreue
 Höchste Diversität auf kleinem Raum anstreben
 Outfits und Emotionen separat clustern für Sub-LoRAs
 Charaktere stets getrennt behandeln eigene Datasets pro Figur



Bonus: Automatisierung

 Nutze `Makefile`, `Bash Script` oder `Python Pipeline`
 Ideal für große Serien mit vielen Charakteren
 Optional: Multi-Char-Pipeline mit auto-sorting nach Klassifikator



 Erste Pipeline-Implementation

 Die Weboberfläche erlaubt Upload eines Videos in den `input/` Ordner
 Ein "Start"-Button löst die Verarbeitung aus
 Nach erfolgreichem Durchlauf wird der komplette `output/` Ordner gezippt und zum Download angeboten
 Bei Fehlern steht die Prozess-Logdatei zum Download bereit
 Jedes Modul protokolliert seine Schritte in `logs/process.log`
 Zwischenstufen arbeiten in eigenen Unterordnern unter `work/` und werden nach Abschluss bereinigt
