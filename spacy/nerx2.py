# Import required libraries
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher


@spacy.Language.component("pokemon_ner")
def pokemon_ner(doc):
    # Create a PhraseMatcher object with the vocabulary from the doc
    matcher = PhraseMatcher(doc.vocab)
    # Tokenize the phrases in the POKEMON_NAMES list
    patterns = list(nlp.tokenizer.pipe(POKEMON_NAMES))
    # Add the patterns to the PhraseMatcher object
    matcher.add("POKEMON_NAMES", None, *patterns)
    # Find all matches in the doc using the PhraseMatcher object
    matches = matcher(doc)
    # Create a new Span object for each match
    spans = [Span(doc, start, end, label="POKEMON") for match_id, start, end in matches]
    # Set the entities of the doc to the new spans
    doc.ents = spans
    # Return the updated doc
    return doc


# Define a list of Pokemon names
POKEMON_NAMES = ['Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle']
# Create a blank spacy model and add the custom component to it
nlp = spacy.blank("en")
nlp.add_pipe("pokemon_ner", name="pokemon_ner")


# Define some text to be processed
poke_txt = "I choose you, Pikachu!"
# Process the text with the spacy model
doc_poke = nlp(poke_txt)
# Print the detected entities and their labels
print([(ent.text, ent.label_) for ent in doc_poke.ents])
# Output
# [('Pikachu', 'POKEMON')]

