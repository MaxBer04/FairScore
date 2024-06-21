import os
import csv
import numpy as np
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_dataset(output_dir):
    # Pfad zur CSV-Datei
    csv_path = os.path.join(output_dir, 'metadata.csv')
    
    # Pfad zum h_vects-Verzeichnis
    h_vects_dir = os.path.join(output_dir, 'h_vects')

    # Analysiere die CSV-Datei
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Anzahl der Elemente im Datensatz
    num_elements = len(rows)

    # Dimensionen eines h_vects-Tensors
    sample_h_vect = np.load(os.path.join(h_vects_dir, rows[0]['h_vects_filename']))
    h_vect_shape = sample_h_vect[list(sample_h_vect.keys())[0]].shape

    # Zusätzliche Informationen
    occupations = Counter(row['occupation'] for row in rows)
    face_detected_count = sum(int(row['face_detected']) for row in rows)
    gender_distribution = Counter(row['prompt'].split()[7] for row in rows)  # 7th word is 'male' or 'female'

    # Ausgabe der Ergebnisse
    print(f"Anzahl der Elemente im Datensatz: {num_elements}")
    print(f"Anzahl der h_vects-Tensoren pro Datenpunkt: {len(sample_h_vect.keys())}")
    print(f"Dimensionen eines h_vects-Tensors: {h_vect_shape}")
    print(f"Anzahl der eindeutigen Berufe: {len(occupations)}")
    print(f"Top 5 häufigste Berufe: {occupations.most_common(5)}")
    print(f"Anzahl der erkannten Gesichter: {face_detected_count} ({face_detected_count/num_elements*100:.2f}%)")
    print(f"Geschlechterverteilung in den Prompts: {dict(gender_distribution)}")

if __name__ == "__main__":
    output_dir = os.path.join(script_dir, "output")  # Passen Sie dies an, wenn Ihr Ausgabeverzeichnis anders heißt
    analyze_dataset(output_dir)