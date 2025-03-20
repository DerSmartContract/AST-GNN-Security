# AST-GNN-Security
# AST-Anomaly-Scanner

Ein **vereinfachtes** Beispielprojekt zur **Anomalieerkennung in Python-Code** mithilfe von:
- **Abstract Syntax Trees (ASTs)** zur Extraktion der Code-Struktur
- **Graph Neural Networks (GNN)** (Autoencoder) zur Erkennung ungewöhnlicher Code-Pfade
- **Streamlit** für eine interaktive Web-UI

> **Achtung**: Dieses Projekt dient als **Demonstration** und **Proof-of-Concept** – das verwendete GNN-Modell ist untrainiert. In einer realen Anwendung solltest du es mit echten Datensätzen trainieren und die Features sowie Modellarchitektur erweitern.

## Funktionsweise

1. **Code-Eingabe**: Du kopierst deinen Python-Code in die Streamlit-UI.
2. **AST-Erstellung**: Mit dem Python-`ast`-Modul wird der Code in einen abstrakten Syntaxbaum überführt.
3. **Graph-Konstruktion**: Ein NetworkX-Graph repräsentiert die Eltern-Kind-Beziehungen im AST.
4. **GNN (Autoencoder)**: Eine einfache, untrainierte GCN-Encoder-Decoder-Architektur versucht, die Knotenfeatures zu rekonstruieren. Der `Reconstruction Error` wird als Indikator für Anomalien interpretiert.
5. **UI-Ausgabe**: Die UI zeigt dir einen geschätzten Fehlerwert; sehr hohe Werte könnten auf ungewöhnliche oder verdächtige Code-Strukturen hinweisen.

## Installation

1. **Repository klonen**:
   ```bash
   git clone https://github.com/DeinBenutzername/AST-Anomaly-Scanner.git
   cd AST-Anomaly-Scanner
