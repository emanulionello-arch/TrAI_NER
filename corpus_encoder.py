import os

def text_cleaner(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        testo = f.read()
        
    testo = testo.replace('“', '"').replace('”', '"')
    testo = testo.replace('‘', "'").replace('’', "'")
    testo = testo.replace('\xa0', ' ')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(testo)

text_cleaner("corpus_grande_grezzo.txt", "corpus_grande_cleaned.txt")