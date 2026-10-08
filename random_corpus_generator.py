import random
import os

professions = ["ingegnere", "architetto", "avvocato", "medico", "presidente", "ministro"]
fem_professions = ["ingegnera", "architetta", "avvocata", "medica", "presidentessa", "ministra"]
names = ["Maria Bianchi", "Laura Rossi", "Francesca Neri", "Elena Verdi", "Giulia Romano", "Sofia Costa", "Chiara Fontana"]
newspapers = ["Corriere della Sera", "La Repubblica", "Il Sole 24 Ore", "La Stampa", "ANSA", "Il Foglio"]
fillers = [
    "Il mercato finanziario ha mostrato segni di ripresa nella giornata di ieri.",
    "Le condizioni meteorologiche rimarranno stabili per tutto il fine settimana.",
    "Il governo ha approvato il nuovo decreto sulla digitalizzazione della pubblica amministrazione.",
    "La squadra locale ha vinto la partita con un gol negli ultimi minuti di gioco.",
    "La ricerca scientifica continua a fare passi da gigante nel campo delle biotecnologie.",
    "Le strade del centro sono state chiuse al traffico per una manifestazione culturale.",
    "Si prevede un aumento della domanda di energia rinnovabile nel prossimo decennio.",
    "La mostra d'arte moderna ha attirato migliaia di visitatori in soli tre giorni."
]

def generate_sentence():
    choice = random.random()
    name = random.choice(names)
    prof = random.choice(professions)
    fem_prof = fem_professions[professions.index(prof)]
    
    if choice < 0.05:
        return f"L'{prof} {name} ha dichiarato che i lavori procederanno secondo i piani."
    elif choice < 0.10:
        return f"L'{fem_prof} {name} ha presentato la sua relazione durante l'assemblea generale."
    else:
        return random.choice(fillers)

file_path = "corpus_test_stress.txt"
target_size_bytes = 110 * 1024 * 1024

with open(file_path, "w", encoding="utf-8") as f:
    current_size = 0
    doc_id = 1
    while current_size < target_size_bytes:
        paper = random.choice(newspapers)
        title = f"Report Speciale numero {doc_id}"
        header = f'<doc newspaper="{paper}" title="{title}">\n'
        f.write(header)
        current_size += len(header.encode("utf-8"))
        
        for _ in range(500):
            sent = generate_sentence() + " "
            f.write(sent)
            current_size += len(sent.encode("utf-8"))
            
        footer = "\n</doc>\n"
        f.write(footer)
        current_size += len(footer.encode("utf-8"))
        doc_id += 1

print(f"File generato: {file_path}")

# Genera un file di testo da 110MB con tag XML fittizi
# Include un 10% di frasi mirate sulle professioni declinate al maschile o femminile
# Il file finale sara pronto nella stessa cartella per essere caricato su Streamlit