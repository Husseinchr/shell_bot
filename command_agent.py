
"""Command Agent Module

Main agent class that:
- Loads and preprocesses linuxcommands.json dataset
- Uses NLP pipeline for feature extraction
- Matches natural language to Linux commands
- Takes care of multi-step commands"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from nlp_pipeline import NLPPipeline
from malware_detector import MalwareDetector


class CommandAgent:
    """AI agent that translates natural language to Linux commands."""
    
    def __init__(self, dataset_path: str = "Dataset/linuxcommands.json"):
        """Sets up the command agent.
        
        Takes in:
            dataset_path: Path to the linuxcommands.json file"""
        self.nlp_pipeline = NLPPipeline()
        self.malware_detector = MalwareDetector()
        self.dataset_path = dataset_path
        

        self.training_data: List[Dict[str, str]] = []
        self.input_texts: List[str] = []
        self.output_commands: List[str] = []
        

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.input_vectors = None
        

        self._load_dataset()
        self._build_model()
    
    def _load_dataset(self):
        """Loads the training dataset from JSON file."""
        dataset_file = Path(self.dataset_path)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        print(f"Loading dataset from {self.dataset_path}...")
        with open(dataset_file, 'r', encoding='Utf-8') as f:
            self.training_data = json.load(f)
        

        self.input_texts = [item['Input'] for item in self.training_data]
        self.output_commands = [item['Output'] for item in self.training_data]
        
        print(f"Loaded {len(self.training_data)} training examples")
    
    def _extract_command_template(self, user_input: str, matched_input: str, matched_output: str) -> str:
        """Pulls out command template and replaces variables with actual values from user input.
        This allows the agent to learn command structures rather than exact matches.
        
        Takes in:
            user_input: Original user input
            matched_input: Matched input from dataset
            matched_output: Matched output command from dataset
        
        Gives back:
            Command with variables replaced from user input"""
        user_lower = user_input.lower()
        matched_lower = matched_input.lower()
        

        user_words = user_input.split()
        matched_words = matched_input.split()
        

        if 'Print' in user_lower and 'Echo' in matched_output.lower():

            print_idx = user_lower.find('Print')
            if print_idx != -1:

                after_print = user_input[print_idx + 5:].strip()

                after_print = re.sub(r'\s+in\s+(the\s+)?terminal', '', after_print, flags=re.IGNORECASE).strip()
                if after_print:

                    if after_print.startswith('"') and after_print.endswith('"'):
                        return f"echo {after_print}"
                    elif after_print.startswith("'") and after_print.endswith("'"):
                        return f"echo {after_print}"
                    else:
                        return f"echo '{after_print}'"
        

        if any(phrase in user_lower for phrase in ['Go to', 'Enter', 'Change directory to', 'Navigate to']):

            dir_name = None
            

            for word in user_words:
                word_clean = word.strip("'\".,;:!?")
                if word_clean and word_clean[0].isupper() and word_clean.isalnum():
                    dir_name = word_clean
                    break
            

            if not dir_name:
                for word in user_words:
                    word_clean = word.strip("'\".,;:!?")
                    if '_' in word_clean:
                        dir_name = word_clean
                        break
            

            if not dir_name:
                for i, word in enumerate(user_words):
                    word_lower = word.lower()
                    if word_lower in ['To', 'Enter', 'Directory'] and i + 1 < len(user_words):
                        next_word = user_words[i + 1].strip("'\".,;:!?")
                        if next_word.lower() not in ['Directory', 'Folder', 'Dir', 'The', 'a', 'An']:
                            dir_name = next_word
                            break
            

            if dir_name:
                if 'Cd' in matched_output.lower():
                    return f'Cd {dir_name}'
        

        if 'Show' in user_lower and 'Directory' in user_lower:
            if 'Pwd' in matched_output.lower() or 'Show' in matched_output.lower():
                return 'Pwd'
        

        if 'Echo' in matched_output.lower() and 'Print' in user_lower:

            quoted_match = re.search(r"echo\s+['\"](.*?)['\"]", matched_output)
            if quoted_match:


                text_to_echo = None
                if 'Print' in user_lower:
                    print_idx = user_lower.find('Print')
                    if print_idx != -1:
                        after_print = user_input[print_idx + 5:].strip()
                        after_print = re.sub(r'\s+in\s+(the\s+)?terminal', '', after_print, flags=re.IGNORECASE).strip()
                        if after_print:
                            text_to_echo = after_print
                
                if text_to_echo:
                    return f"echo '{text_to_echo}'"
        

        return matched_output
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for matching using NLP pipeline.
        preserves important information while normalizing the text.
        emphasizes action verbs to improve matching accuracy.
        Takes in:
        text: Input text
        Gives back:
        preprocessed text string optimized for TF-IDF matching"""

        keywords = self.nlp_pipeline.get_keywords(text, filter_stopwords=True)
        

        result = self.nlp_pipeline.process(text)
        doc = result['Doc']
        


        important_lemmas = []
        verbs = []  # Track verbs separately for emphasis
        nouns = []  # Track nouns for better context
        
        for token in doc:


            if token.pos_ == 'VERB':
                lemma = token.lemma_.lower()
                important_lemmas.append(lemma)
                verbs.append(lemma)

                if lemma not in ['Be', 'Have', 'Do']:  # Skip auxiliary verbs
                    verbs.append(lemma)
            elif token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop:
                lemma = token.lemma_.lower()
                important_lemmas.append(lemma)
                if token.pos_ == 'NOUN':
                    nouns.append(lemma)
        

        chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        


        filtered_chunks = []
        for chunk in chunks:

            if '/dev/' not in chunk and not chunk.startswith('/sys') and not chunk.startswith('/proc'):
                filtered_chunks.append(chunk)
        



        processed_parts = keywords + important_lemmas + (verbs * 2) + nouns + filtered_chunks
        

        seen = set()
        unique_parts = []
        for part in processed_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        

        processed_text = ' '.join(unique_parts)
        
        return processed_text if processed_text else text.lower()
    
    def _build_model(self):
        """Builds the TF-IDF model for semantic matching."""
        print("Building TF-IDF model...")
        

        processed_inputs = [self._preprocess_text(text) for text in self.input_texts]
        


        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        action_verbs_to_keep = {'Go', 'Do', 'Make', 'Get', 'Set', 'Run', 'Show', 'List', 'Find', 'Copy', 'Move', 'Delete', 'Remove', 'Create', 'Change', 'Enter', 'Back', 'Previous'}
        custom_stop_words = list(ENGLISH_STOP_WORDS - action_verbs_to_keep)
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased to capture more patterns
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams for better phrase matching
            min_df=1,  # Lower threshold to capture more unique patterns
            max_df=0.9,  # Slightly lower to reduce common word dominance
            stop_words=custom_stop_words,  # Custom stop words that preserve action verbs
            analyzer='Word',  # Word-level analysis
            lowercase=True,
            norm='l2'  # L2 normalization for cosine similarity
        )
        

        self.input_vectors = self.vectorizer.fit_transform(processed_inputs)
        
        print(f"Model built with {self.input_vectors.shape[1]} features")
    
    def translate(self, user_input: str, top_k: int = 3) -> str:
        """Translate natural language input to Linux command(s).
        Takes in:
        user_input: Natural language prompt
        top_k: Number of top matches to consider
        Gives back:
        linux command(s) or "echo 'Malware detected'" if dangerous"""

        if self.malware_detector.is_malware(user_input):
            return "echo 'Malware detected'"
        

        if self._is_direct_command(user_input):
            return user_input
        

        if self._is_multi_step(user_input):
            return self._handle_multi_step(user_input)
        


        short_command_match = self._handle_short_commands(user_input)
        if short_command_match:
            return short_command_match
        

        best_match = self._find_best_match(user_input, top_k)
        
        if best_match:
            return best_match
        else:

            return f"echo 'Command not found for: {user_input}'"
    
    def _is_direct_command(self, text: str) -> bool:
        """Checks whether input is already a direct command (not natural language).
        Takes in:
        text: Input text
        Gives back:
        true if it looks like a direct command"""

        natural_language_starters = [
            'I want', 'i need', 'Please', 'Can you', 'Could you',
            'How do i', 'How to', 'Show me', 'Help me'
        ]
        
        text_lower = text.lower().strip()
        

        if any(text_lower.startswith(starter) for starter in natural_language_starters):
            return False
        

        natural_phrases = [
            'Go back', 'Go to', 'Go home', 'Go previous', 'Navigate back',
            'Print hello', 'Print', 'Show current', 'Show directory', 'Show the directory',
            'List all', 'List files', 'Copy files', 'Copy all', 'Enter directory', 'Change directory'
        ]
        if text_lower in natural_phrases or any(text_lower.startswith(phrase + ' ') for phrase in natural_phrases):
            return False
        


        command_patterns = [
            r'^[a-z]+\s+[-/]',  # Command with flag
            r'^[a-z]+\s+/',  # Command with absolute path
            r'^[a-z]+\s+\./',  # Command with relative path
        ]
        
        if any(re.match(pattern, text_lower) for pattern in command_patterns):

            if not any(marker in text_lower for marker in ['The', 'a', 'An', 'This', 'That']):
                return True
        

        if re.match(r'^[a-z]+\s+[a-z]+$', text_lower):

            if text_lower not in natural_phrases:

                if not any(marker in text_lower for marker in ['The', 'a', 'An', 'This', 'That', 'Your', 'My']):
                    return True
        
        return False
    
    def _handle_short_commands(self, user_input: str) -> Optional[str]:
        """Takes care of short common commands that might not match well with TF-IDF
        due to stopword filtering. Uses exact matching on dataset.
        Takes in:
        user_input: User input text
        Gives back:
        matched command or None"""
        user_lower = user_input.lower().strip()
        user_original = user_input.strip()
        

        words = user_input.split()
        if len(words) <= 3:

            user_words_normalized = [w.lower().strip() for w in words]
            for idx, training_input in enumerate(self.input_texts):
                training_words_normalized = [w.lower().strip() for w in training_input.split()]

                if user_words_normalized == training_words_normalized:
                    return self.output_commands[idx]
            

            for idx, training_input in enumerate(self.input_texts):
                training_words_lower = [w.lower() for w in training_input.split()]

                if len(training_words_lower) >= len(user_words_normalized) and len(training_words_lower) <= len(user_words_normalized) + 1:

                    user_idx = 0
                    for train_word in training_words_lower:
                        if user_idx < len(user_words_normalized) and train_word == user_words_normalized[user_idx]:
                            user_idx += 1

                    if user_idx == len(user_words_normalized):

                        training_lower = training_input.lower()
                        if any(cmd in training_lower for cmd in ['Go back', 'Go to', 'Go home', 'Go previous', 'Previous', 'Back', 'Print hello', 'Print', 'Echo hello']):
                            return self.output_commands[idx]
        
        return None
    
    def _is_multi_step(self, text: str) -> bool:
        """Checks whether input contains multiple sequential actions.
        Takes in:
        text: Input text
        Gives back:
        true if multi-step detected"""

        sequential_markers = [
            'Then', 'And then', 'After that', 'After', 'Next',
            'Followed by', 'Subsequently', 'Afterwards'
        ]
        
        text_lower = text.lower()
        

        for marker in sequential_markers:
            if marker in text_lower:
                return True
        

        result = self.nlp_pipeline.process(text)
        verbs = [token for token in result['Pos_tags'] if token['Pos'] == 'VERB']
        

        if len(verbs) >= 2:

            conjunctions = ['And', 'Then', 'After', 'Next']
            if any(conj in text_lower for conj in conjunctions):
                return True
        
        return False
    
    def _handle_multi_step(self, text: str) -> str:
        """Takes care of multi-step commands by splitting and translating each step.
        Takes in:
        text: Multi-step natural language input
        Gives back:
        multiple commands separated by semicolons"""


        text_lower = text.lower().strip()
        for idx, training_input in enumerate(self.input_texts):
            if training_input.lower().strip() == text_lower:
                return self.output_commands[idx]
        


        sequential_markers = [
            ' then ', ' and then ', ' after that ', ' next ',
            ' followed by ', ' subsequently ', ' afterwards '
        ]
        

        steps = [text]
        for marker in sequential_markers:
            new_steps = []
            for step in steps:
                if marker in step.lower():
                    parts = re.split(marker, step, flags=re.IGNORECASE)
                    new_steps.extend(parts)
                else:
                    new_steps.append(step)
            steps = new_steps
        


        final_after_steps = []
        for step in steps:
            if ' after ' in step.lower():


                if any(phrase in step.lower() for phrase in ['After showing', 'After listing', 'After displaying', 'After printing']):
                    final_after_steps.append(step)
                else:

                    parts = re.split(r'\s+after\s+', step, flags=re.IGNORECASE, maxsplit=1)
                    if len(parts) == 2:

                        first_has_verb = any(word in parts[0].lower() for word in ['List', 'Show', 'Display', 'Print', 'Copy', 'Move', 'Delete'])
                        second_has_verb = any(word in parts[1].lower() for word in ['Showing', 'Listing', 'Displaying', 'Printing', 'Copying', 'Moving'])
                        if first_has_verb and second_has_verb:
                            final_after_steps.extend(parts)
                        else:
                            final_after_steps.append(step)
                    else:
                        final_after_steps.append(step)
            else:
                final_after_steps.append(step)
        steps = final_after_steps
        


        final_steps = []
        for step in steps:

            step_lower = step.lower().strip()
            for idx, training_input in enumerate(self.input_texts):
                if training_input.lower().strip() == step_lower:

                    final_steps = [step]  # Keep original step for template matching
                    break


            step_lower = step.lower()
            if any(phrase in step_lower for phrase in ['After showing', 'After listing', 'Then showing', 'Then listing']):


                exact_match = None
                for idx, training_input in enumerate(self.input_texts):
                    if training_input.lower().strip() == step.lower().strip():
                        exact_match = self.output_commands[idx]
                        break
                
                if exact_match:
                    final_steps.append(step)
                    continue
            

            result = self.nlp_pipeline.process(step)
            verbs = [token for token in result['Pos_tags'] if token['Pos'] == 'VERB']
            
            if len(verbs) >= 2 and ' and ' in step.lower():


                if not any(phrase in step_lower for phrase in ['After', 'Before', 'While', 'When']):

                    parts = re.split(r'\s+and\s+', step, flags=re.IGNORECASE, maxsplit=1)
                    if len(parts) == 2:

                        second_result = self.nlp_pipeline.process(parts[1])
                        second_verbs = [token for token in second_result['Pos_tags'] if token['Pos'] == 'VERB']
                        if second_verbs:
                            final_steps.extend(parts)
                            continue
            
            final_steps.append(step)
        

        commands = []
        for step in final_steps:
            step = step.strip()
            if step:

                exact_match = None
                for idx, training_input in enumerate(self.input_texts):
                    if training_input.lower().strip() == step.lower().strip():
                        exact_match = self.output_commands[idx]
                        break
                
                if exact_match:
                    commands.append(exact_match)
                    continue
                


                command = self._find_best_match(step, top_k=5)
                

                if not command or command.startswith("echo 'Command not found"):
                    short_match = self._handle_short_commands(step)
                    if short_match:
                        command = short_match
                    else:

                        command = self._find_best_match(step, top_k=1)
                
                if command and not command.startswith("echo 'Command not found"):
                    commands.append(command)
        

        if commands:
            return '; '.join(commands)
        else:
            return self._find_best_match(text, top_k=1) or f"echo 'Command not found for: {text}'"
    
    def _rule_based_match(self, user_input: str) -> Optional[str]:
        """Rule-based matching for common commands before using TF-IDF.
        this handles common patterns more accurately.
        Takes in:
        user_input: Natural language input
        Gives back:
        matched command or None"""
        text_lower = user_input.lower().strip()
        

        result = self.nlp_pipeline.process(user_input)
        doc = result['Doc']
        

        directory_names = []
        file_names = []
        common_words = {'Directory', 'Folder', 'File', 'Current', 'This', 'That', 'The', 'a', 'An'}
        
        for token in doc:
            if token.pos_ == 'NOUN' and token.text.lower() not in common_words:


                token_idx = token.i
                context_start = max(0, token_idx - 3)
                context_end = min(len(doc), token_idx + 3)
                context_text = ' '.join([t.text.lower() for t in doc[context_start:context_end]])
                
                if any(word in context_text for word in ['Directory', 'Folder', 'Dir', 'In the', 'In']):
                    directory_names.append(token.text)
                else:

                    file_names.append(token.text)
        

        words = user_input.split()
        for i, word in enumerate(words):

            if '/' in word:
                directory_names.append(word)

            elif '_' in word and word.lower() not in common_words:


                context_start = max(0, i-3)
                context_end = min(len(words), i+3)
                context = ' '.join(words[context_start:context_end]).lower()
                if any(marker in context for marker in ['To', 'Into', 'Directory', 'Folder', 'Change', 'Enter', 'Go', 'The', 'In the', 'In']):
                    directory_names.append(word)

            elif word.replace('_', '').isalnum() and len(word) > 2 and word.lower() not in common_words:

                context_start = max(0, i-3)
                context_end = min(len(words), i+3)
                context = ' '.join(words[context_start:context_end]).lower()
                

                if i > 0 and words[i-1].lower() in ['Folder', 'Directory', 'Dir']:
                    directory_names.append(word)

                elif i < len(words) - 1 and words[i+1].lower() in ['Folder', 'Directory', 'Dir']:
                    directory_names.append(word)

                elif any(marker in context for marker in ['To', 'Into', 'Change', 'Enter', 'Go']) and word.lower() not in ['Files', 'File', 'Items', 'Content', 'List', 'Show', 'Display']:
                    directory_names.append(word)
        

        directory_names = list(dict.fromkeys(directory_names))
        file_names = list(dict.fromkeys(file_names))
        

        prioritized_dirs = []
        other_dirs = []
        
        words = user_input.split()
        for dir_name in directory_names:

            if '_' in dir_name:
                prioritized_dirs.append(dir_name)

            elif dir_name in words:
                idx = words.index(dir_name)
                if idx < len(words) - 1 and words[idx + 1].lower() in ['Folder', 'Directory', 'Dir']:
                    prioritized_dirs.append(dir_name)
                else:
                    other_dirs.append(dir_name)
            else:
                other_dirs.append(dir_name)
        

        directory_names = prioritized_dirs + [d for d in other_dirs if d not in prioritized_dirs]
        

        if any(word in text_lower for word in ['List', 'Show', 'Display']) and any(word in text_lower for word in ['File', 'Files', 'Content', 'Items']):

            if directory_names:
                dir_name = directory_names[0]
                if 'Detail' in text_lower or 'All' in text_lower or 'Hidden' in text_lower:
                    return f'Ls -la {dir_name}'
                elif 'Recursive' in text_lower or 'Subdirectory' in text_lower:
                    return f'Ls -laR {dir_name}'
                else:
                    return f'Ls -la {dir_name}'
            elif 'Detail' in text_lower or 'All' in text_lower or 'Hidden' in text_lower:
                return 'Ls -la'
            elif 'Recursive' in text_lower or 'Subdirectory' in text_lower:
                return 'Ls -laR'
            else:
                return 'Ls -la'
        


        cd_patterns = [
            'Change directory', 'Go to', 'Navigate to', 'Switch to', 'Move to', 
            'Enter directory', 'Enter the directory', 'Cd'
        ]

        starts_with_enter = text_lower.startswith('Enter ')
        
        if any(pattern in text_lower for pattern in cd_patterns) or \
           (starts_with_enter and (directory_names or 'Directory' in text_lower)):
            if directory_names:

                dir_name = directory_names[0]
                return f'Cd {dir_name}'
            elif starts_with_enter and not directory_names:

                words_after_enter = user_input.split()[1:]  # Skip "enter"
                if words_after_enter:

                    potential_dir = words_after_enter[0]

                    if potential_dir.lower() not in ['The', 'a', 'An', 'Directory', 'Folder', 'Dir']:
                        return f'Cd {potential_dir}'
            elif 'Home' in text_lower:
                return 'Cd ~'
            elif 'Root' in text_lower:
                return 'Cd /'
            elif 'Previous' in text_lower or 'Back' in text_lower:
                return 'Cd -'
            elif 'Up' in text_lower or 'Parent' in text_lower:
                return 'Cd ..'
            else:
                return 'Cd'
        


        if any(word in text_lower for word in ['Current directory', 'Working directory', 'Where am i', 'Pwd', 'Show directory']):
            return 'Pwd'

        if 'Show' in text_lower and ('Current' in text_lower or 'Working' in text_lower) and ('Directory' in text_lower or 'Folder' in text_lower):
            return 'Pwd'
        

        if 'Copy' in text_lower or 'Cp' in text_lower:
            if file_names and directory_names:
                return f'Cp {file_names[0]} {directory_names[0]}'
            elif file_names:
                return f'Cp {file_names[0]}'
            else:
                return 'Cp'
        

        if 'Move' in text_lower or ('Mv' in text_lower and 'Move' in text_lower):
            if file_names and directory_names:
                return f'Mv {file_names[0]} {directory_names[0]}'
            elif file_names:
                return f'Mv {file_names[0]}'
            else:
                return 'Mv'
        

        if any(word in text_lower for word in ['Create directory', 'Make directory', 'Mkdir', 'New folder']):
            if directory_names:
                return f'Mkdir {directory_names[0]}'
            else:
                return 'Mkdir'
        

        if any(word in text_lower for word in ['Remove', 'Delete', 'Rm']) and not self.malware_detector.is_malware(user_input):
            if file_names:
                return f'Rm {file_names[0]}'
            elif directory_names:
                return f'Rm -r {directory_names[0]}'
            else:
                return 'Rm'
        
        return None
    
    def _find_best_match(self, user_input: str, top_k: int = 3) -> Optional[str]:
        """Find the best matching command for user input using TF-IDF.
        Takes in:
        user_input: Natural language input
        top_k: Number of top matches to consider
        Gives back:
        best matching command or None"""
        if not self.vectorizer or self.input_vectors is None:
            return None
        

        processed_input = self._preprocess_text(user_input)
        

        try:
            user_vector = self.vectorizer.transform([processed_input])
        except:

            return self._simple_match(user_input)
        

        similarities = cosine_similarity(user_vector, self.input_vectors)[0]
        

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        


        best_idx = top_indices[0]
        best_similarity = similarities[best_idx]
        

        user_words = set(user_input.lower().split())
        user_processed = self._preprocess_text(user_input)
        tied_indices = [idx for idx in top_indices if abs(similarities[idx] - best_similarity) < 0.001]
        
        if len(tied_indices) > 1:

            for idx in tied_indices:
                if self.input_texts[idx].lower().strip() == user_input.lower().strip():
                    best_idx = idx
                    break
            else:

                best_word_overlap = -1
                best_action_match = False
                

                user_result = self.nlp_pipeline.process(user_input)
                user_verbs = {token.lemma_.lower() for token in user_result['Doc'] if token.pos_ == 'VERB'}
                
                for idx in tied_indices:
                    match_words = set(self.input_texts[idx].lower().split())
                    overlap = len(user_words & match_words)
                    

                    length_diff = abs(len(self.input_texts[idx].split()) - len(user_input.split()))
                    

                    match_result = self.nlp_pipeline.process(self.input_texts[idx])
                    match_verbs = {token.lemma_.lower() for token in match_result['Doc'] if token.pos_ == 'VERB'}
                    action_match = len(user_verbs & match_verbs) > 0
                    

                    score = (action_match * 1000) + (100 - length_diff) + overlap
                    
                    if score > (best_action_match * 1000 + 100 - abs(len(self.input_texts[best_idx].split()) - len(user_input.split())) + best_word_overlap):
                        best_action_match = action_match
                        best_word_overlap = overlap
                        best_idx = idx
        



        word_count = len(user_input.split())
        if word_count <= 3:
            threshold = 0.1  # Lower threshold for short commands like "go back"
        else:
            threshold = 0.15  # Standard threshold for longer commands
        
        if best_similarity > threshold:
            matched_command = self.output_commands[best_idx]
            matched_input = self.input_texts[best_idx]
            

            template_command = self._extract_command_template(user_input, matched_input, matched_command)
            


            improved_command = self._improve_command_with_context(user_input, template_command, best_similarity)
            
            return improved_command
        else:

            return self._simple_match(user_input)
    
    def _improve_command_with_context(self, user_input: str, command: str, similarity: float = 0.0) -> str:
        """Try to improve a matched command by extracting exact directory/file names from user input
        and replacing them in the matched command. Preserves case sensitivity.
        Takes in:
        user_input: Original user input
        command: Matched command from dataset
        similarity: Similarity score of the match
        Gives back:
        improved command with exact names from user input"""


        user_words = user_input.split()
        exact_dir_names = []
        directory_names = []
        file_names = []
        common_words = {'Directory', 'Folder', 'File', 'Files', 'Current', 'This', 'That', 'The', 'a', 'An', 'To', 'From', 'In', 'On', 'At'}
        action_verbs = {'Enter', 'Change', 'Go', 'Move', 'Switch', 'Navigate', 'List', 'Show', 'Display', 'Copy', 'Cp', 'Move', 'Mv', 'Delete', 'Remove', 'Rm'}
        
        for i, word in enumerate(user_words):
            word_clean = word.strip("'\".,;:!?")
            if not word_clean:  # Skip empty words
                continue
            word_lower = word_clean.lower()
            

            if word_lower in action_verbs:
                continue
            

            if word_lower in common_words and not (word_clean[0].isupper() or '_' in word_clean):
                continue
            

            prev_word = user_words[i-1].lower() if i > 0 else ''
            prev_words = ' '.join(user_words[max(0, i-3):i]).lower()
            next_word = user_words[i+1].lower() if i+1 < len(user_words) else ''
            


            if word_clean[0].isupper() and word_clean.isalnum() and len(word_clean) > 2:

                if prev_word == 'To' or next_word in ['Directory', 'Folder', 'Dir'] or \
                   any(ind in prev_words for ind in ['To', 'Into', 'Change', 'Enter', 'Go', 'From']):
                    exact_dir_names.append(word_clean)

            elif '_' in word_clean and word_clean.replace('_', '').isalnum():
                if prev_word == 'To' or next_word in ['Directory', 'Folder', 'Dir'] or \
                   any(ind in prev_words for ind in ['To', 'Into', 'Change', 'Enter', 'Go', 'From']):
                    exact_dir_names.append(word_clean)

            elif word_clean.isalnum() and len(word_clean) > 2 and not word_clean[0].isupper():
                if (prev_word in ['Enter', 'To'] or 'Change directory to' in prev_words) and \
                   (next_word in ['Directory', 'Folder', 'Dir'] or i == len(user_words) - 1):
                    exact_dir_names.append(word_clean)
        

        words = user_input.split()
        for i, word in enumerate(words):
            word_clean = word.strip("'\".,;:!?")
            word_lower = word_clean.lower()
            

            if word_clean in exact_dir_names or word_lower in common_words:
                continue
            

            prev_context = ' '.join(words[max(0, i-3):i]).lower()
            next_context = ' '.join(words[i+1:min(len(words), i+4)]).lower()
            full_context = prev_context + ' ' + next_context
            
            dir_indicators = ['Directory', 'Folder', 'Dir', 'To', 'Into', 'Change', 'Enter', 'Go', 'From']
            file_indicators = ['File', 'Files']
            
            is_directory = any(indicator in full_context for indicator in dir_indicators)
            is_file = any(indicator in full_context for indicator in file_indicators)
            

            if '/' in word_clean:
                directory_names.append(word_clean)

            elif word_clean.isalnum() and len(word_clean) > 2 and not word_clean[0].isupper() and '_' not in word_clean:
                if is_directory:
                    directory_names.append(word_clean)
                elif is_file:
                    file_names.append(word_clean)
        


        if exact_dir_names:
            directory_names = exact_dir_names + [d for d in directory_names if d.lower() not in [ed.lower() for ed in exact_dir_names]]
        else:

            seen_lower = set()
            unique_dirs = []
            for dir_name in directory_names:
                if dir_name.lower() not in seen_lower:
                    seen_lower.add(dir_name.lower())
                    unique_dirs.append(dir_name)
            directory_names = unique_dirs
        

        user_lower = user_input.lower()
        

        if command.startswith('Cd '):

            if 'Previous' in user_lower or ('Back' in user_lower and ('Go' in user_lower or 'Navigate' in user_lower)):
                if 'Previous directory' in user_lower or 'Go back' in user_lower or 'Navigate back' in user_lower or 'Change to previous' in user_lower:
                    return 'Cd -'
            if '~' in user_input or user_input.strip() == 'Go to ~':
                return 'Cd ~'
            if 'Home' in user_lower and ('Directory' in user_lower or 'Go to' in user_lower or 'Navigate' in user_lower or user_input.strip() == 'Go to home'):
                return 'Cd ~'
            if '.' in user_input and ('To .' in user_lower or 'To current' in user_lower):
                return 'Cd .'
        


        if directory_names:
            user_dir = directory_names[0]  # Use the first (most prioritized) directory name
            

            if user_dir.upper() == 'HOME' and 'Home' in user_lower:
                user_dir = '~'
            elif user_dir.lower() == 'Current' or ('.' in user_input and user_dir == '.'):
                user_dir = '.'
            
            if command.startswith('Cd '):

                return f'Cd {user_dir}'
            elif command.startswith('Ls'):

                cmd_parts = command.split()
                if len(cmd_parts) == 1:
                    return f'{command} {user_dir}'
                elif len(cmd_parts) >= 2:

                    return f'{cmd_parts[0]} {user_dir}'
            elif command.startswith('Cp '):

                if 'From' in user_lower and 'To' in user_lower:
                    from_idx = user_lower.find('From')
                    to_idx = user_lower.find('To')
                    if from_idx < to_idx and len(directory_names) >= 1:
                        source = directory_names[0]

                        dest = '.'
                        if 'Home' in user_lower[to_idx:] or 'HOME' in user_input[to_idx:]:
                            dest = '~'
                        elif '.' in user_input[to_idx:] or 'Current' in user_lower[to_idx:] or 'Here' in user_lower[to_idx:]:
                            dest = '.'
                        elif 'Parent' in user_lower[to_idx:] or '..' in user_input[to_idx:]:
                            dest = '..'
                        elif len(directory_names) >= 2:
                            dest = directory_names[1]
                        



                        return f'Cp {source}/* {dest}'
        

        if similarity > 0.8:
            return command
        

        if similarity >= 0.3 and similarity <= 0.7:

            if command.startswith('Cd ') and directory_names:

                return f'Cd {directory_names[0]}'
            elif command.startswith('Ls') and directory_names:

                cmd_parts = command.split()
                if len(cmd_parts) == 1 or (len(cmd_parts) == 2 and cmd_parts[1] not in directory_names):
                    return f'{command} {directory_names[0]}'
            elif command.startswith('Cp ') and directory_names:

                user_lower = user_input.lower()
                if 'From' in user_lower and 'To' in user_lower:

                    from_idx = user_lower.find('From')
                    to_idx = user_lower.find('To')
                    if from_idx < to_idx:
                        source = directory_names[0] if directory_names else 'Source'
                        dest = '.' if '.' in user_input else (directory_names[1] if len(directory_names) > 1 else '.')
                        return f'Cp {source}/* {dest}'
        
        return command
    
    def _simple_match(self, user_input: str) -> Optional[str]:
        """Simple keyword-based matching as fallback.
        Takes in:
        user_input: Natural language input
        Gives back:
        best matching command or None"""

        keywords = self.nlp_pipeline.get_keywords(user_input, filter_stopwords=True)
        
        if not keywords:
            return None
        

        best_match_idx = -1
        best_match_count = 0
        
        for idx, training_input in enumerate(self.input_texts):
            training_keywords = self.nlp_pipeline.get_keywords(training_input, filter_stopwords=True)
            

            common = set(keywords) & set(training_keywords)
            if len(common) > best_match_count:
                best_match_count = len(common)
                best_match_idx = idx
        
        if best_match_idx >= 0 and best_match_count > 0:
            return self.output_commands[best_match_idx]
        
        return None


if __name__ == "__main__":

    print("Initializing Command Agent...")
    agent = CommandAgent()
    
    test_cases = [
        "I want to list all files in this directory with file details",
        "Compress the file data.txt using bzip2",
        "I want to enter the directory data then list the files in it and in every subdirectory of it then copy a specific file to another directory called temp",
        "Show me the current directory",
        "Delete all files",  # Should be caught by malware detector
    ]
    
    print("\nTesting Command Agent")
    print("=" * 60)
    
    for test_input in test_cases:
        print(f"\nInput: {test_input}")
        output = agent.translate(test_input)
        print(f"Output: {output}")

