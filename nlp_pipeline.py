
"""NLP Pipeline Module

Implements the 5-step NLP processing pipeline:
1. Tokenization/Segmentation (NLTK)
2. Stemming/Lemmatization (spaCy)
3. Part-of-Speech Tagging (spaCy)
4. Grammatical Relations/Chunking/Parsing (spaCy)
5. Word Meaning/Text Meaning (WordNet)"""

import re
import nltk
import spacy
from typing import List, Dict, Any, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


try:
    nltk.data.find('Tokenizers/punkt_tab')
except LookupError:
    nltk.download('Punkt_tab', quiet=True)

try:
    nltk.data.find('Corpora/wordnet')
except LookupError:
    nltk.download('Wordnet', quiet=True)

try:
    nltk.data.find('Corpora/omw-1.4')
except LookupError:
    nltk.download('Omw-1.4', quiet=True)

try:
    nltk.data.find('Corpora/stopwords')
except LookupError:
    nltk.download('Stopwords', quiet=True)


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'En_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None


class NLPPipeline:
    """NLP Pipeline class implementing all 5 steps of NLP processing."""
    
    def __init__(self):
        """Sets up the NLP pipeline."""
        self.stop_words = set(stopwords.words('English'))
        if nlp is None:
            raise RuntimeError("spaCy model not loaded. Please install en_core_web_sm")
        self.nlp = nlp
    
    def process(self, text: str) -> Dict[str, Any]:
        """Handles text through all 5 NLP steps.
        
        Takes in:
            text: Input text to process
        
        Gives back:
            Dictionary containing results from all 5 steps"""
        if not text or not text.strip():
            return self._empty_result()
        

        tokens, sentences = self.tokenize(text)
        

        doc = self.nlp(text)
        

        lemmas = self.lemmatize(doc)
        

        pos_tags = self.pos_tag(doc)
        

        dependencies, chunks = self.parse_and_chunk(doc)
        

        word_meanings = self.extract_word_meanings(tokens)
        
        return {
            'Original_text': text,
            'Tokens': tokens,
            'Sentences': sentences,
            'Lemmas': lemmas,
            'Pos_tags': pos_tags,
            'Dependencies': dependencies,
            'Chunks': chunks,
            'Word_meanings': word_meanings,
            'Doc': doc  # Keep spaCy doc for further processing
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Gives back empty result structure."""
        return {
            'Original_text': '',
            'Tokens': [],
            'Sentences': [],
            'Lemmas': [],
            'Pos_tags': [],
            'Dependencies': [],
            'Chunks': [],
            'Word_meanings': [],
            'Doc': None
        }
    

    def tokenize(self, text: str) -> Tuple[List[str], List[str]]:
        """Step 1: Tokenization and Segmentation.
        uses NLTK to:
        - Tokenize text into words
        - Segment text into sentences
        Takes in:
        text: Input text
        Gives back:
        tuple of (word_tokens, sentences)"""

        sentences = sent_tokenize(text)
        

        tokens = word_tokenize(text)
        


        return tokens, sentences
    

    def lemmatize(self, doc) -> List[Dict[str, str]]:
        """Step 2: Lemmatization using spaCy.
        Takes in:
        doc: spaCy document object
        Gives back:
        list of dictionaries with token and lemma"""
        lemmas = []
        for token in doc:
            lemmas.append({
                'Token': token.text,
                'Lemma': token.lemma_,
                'Lower_lemma': token.lemma_.lower()
            })
        return lemmas
    

    def pos_tag(self, doc) -> List[Dict[str, str]]:
        """Step 3: Part-of-Speech Tagging using spaCy.
        Takes in:
        doc: spaCy document object
        Gives back:
        list of dictionaries with token, POS, and detailed tag"""
        pos_tags = []
        for token in doc:
            pos_tags.append({
                'Token': token.text,
                'Pos': token.pos_,  # Universal POS tag
                'Tag': token.tag_,  # Detailed POS tag (Penn Treebank)
                'Is_stop': token.is_stop,
                'Is_alpha': token.is_alpha
            })
        return pos_tags
    

    def parse_and_chunk(self, doc) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Step 4: Dependency Parsing and Chunking using spaCy.
        Takes in:
        doc: spaCy document object
        Gives back:
        tuple of (dependencies, noun_chunks)"""
        dependencies = []
        for token in doc:
            dependencies.append({
                'Token': token.text,
                'Dep': token.dep_,  # Dependency relation
                'Head': token.head.text,  # Head token
                'Head_pos': token.head.pos_,  # Head POS
                'Children': [child.text for child in token.children]
            })
        

        chunks = [chunk.text for chunk in doc.noun_chunks]
        
        return dependencies, chunks
    

    def extract_word_meanings(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Step 5: Pulls out word meanings using WordNet.
        Takes in:
        tokens: List of word tokens
        Gives back:
        list of dictionaries with word meanings and synsets"""
        word_meanings = []
        
        for token in tokens:

            if not token.isalpha() or token.lower() in self.stop_words:
                continue
            
            token_lower = token.lower()
            synsets = wn.synsets(token_lower)
            
            if synsets:

                primary_synset = synsets[0]
                word_meanings.append({
                    'Word': token,
                    'Synsets': [syn.name() for syn in synsets[:3]],  # Top 3 synsets
                    'Definition': primary_synset.definition(),
                    'Examples': primary_synset.examples()[:2] if primary_synset.examples() else []
                })
        
        return word_meanings
    
    def get_keywords(self, text: str, filter_stopwords: bool = True) -> List[str]:
        """Pulls out keywords from text using NLP pipeline.
        Takes in:
        text: Input text
        filter_stopwords: Whether to filter stopwords
        Gives back:
        list of keyword lemmas"""
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:

            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                if not filter_stopwords or not token.is_stop:
                    keywords.append(token.lemma_.lower())
        
        return keywords
    
    def get_entities(self, text: str) -> List[Dict[str, str]]:
        """Pulls out named entities from text.
        Takes in:
        text: Input text
        Gives back:
        list of named entities with their types"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'Text': ent.text,
                'Label': ent.label_,
                'Start': ent.start_char,
                'End': ent.end_char
            })
        
        return entities
    
    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity between two words using WordNet.
        Takes in:
        word1: First word
        word2: Second word
        Gives back:
        similarity score (0.0 to 1.0)"""
        synsets1 = wn.synsets(word1.lower())
        synsets2 = wn.synsets(word2.lower())
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_similarity = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                try:
                    similarity = syn1.path_similarity(syn2)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
                except:
                    continue
        
        return max_similarity



def process_text(text: str) -> Dict[str, Any]:
    """Convenience function to process text through NLP pipeline.
    Takes in:
        text: Input text
    Gives back:
        dictionary with NLP processing results"""
    pipeline = NLPPipeline()
    return pipeline.process(text)


if __name__ == "__main__":

    test_text = "I want to list all files in this directory with file details"
    
    print("Testing NLP Pipeline")
    print("=" * 60)
    print(f"Input: {test_text}\n")
    
    pipeline = NLPPipeline()
    result = pipeline.process(test_text)
    
    print("Step 1 - Tokenization:")
    print(f"  Tokens: {result['Tokens']}")
    print(f"  Sentences: {result['Sentences']}\n")
    
    print("Step 2 - Lemmatization:")
    for lemma_info in result['Lemmas'][:5]:
        print(f"  {lemma_info['Token']} -> {lemma_info['Lemma']}")
    print()
    
    print("Step 3 - POS Tagging:")
    for pos_info in result['Pos_tags'][:5]:
        print(f"  {pos_info['Token']}: {pos_info['Pos']} ({pos_info['Tag']})")
    print()
    
    print("Step 4 - Dependency Parsing:")
    for dep_info in result['Dependencies'][:5]:
        print(f"  {dep_info['Token']} <-{dep_info['Dep']}- {dep_info['Head']}")
    print(f"  Noun Chunks: {result['Chunks']}\n")
    
    print("Step 5 - Word Meaning:")
    for meaning in result['Word_meanings'][:3]:
        print(f"  {meaning['Word']}: {meaning['Definition']}")
    print()

