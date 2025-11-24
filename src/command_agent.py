
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

from .nlp_pipeline import NLPPipeline
from .malware_detector import MalwareDetector


class CommandAgent:
    """AI agent that translates natural language to Linux commands."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """Sets up the command agent.
        
        Takes in:
            dataset_path: Path to the linuxcommands.json file"""
        self.nlp_pipeline = NLPPipeline()
        self.malware_detector = MalwareDetector()
        if dataset_path is None:
            project_root = Path(__file__).parent.parent
            self.dataset_path = str(project_root / "Dataset" / "linuxcommands.json")
        else:
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
        

        self.input_texts = [item.get('input', item.get('Input', '')) for item in self.training_data]
        self.output_commands = [item.get('output', item.get('Output', '')) for item in self.training_data]
        
        print(f"Loaded {len(self.training_data)} training examples")
    
    def _extract_command_template(self, user_input: str, matched_input: str, matched_output: str) -> str:
        """Extract command template and replace placeholders with actual values from user input.
        Intelligently extracts source and destination from "from X to Y" patterns.
        This learns patterns: "list files in [DIR]" → "ls [DIR]" regardless of directory name.
        
        Takes in:
            user_input: Original user input
            matched_input: Matched input from dataset
            matched_output: Matched output command from dataset
        
        Gives back:
            Command with actual directory/file names extracted from user input"""
        
        user_lower = user_input.lower()
        user_words = user_input.split()
        
        if 'copy' in user_lower or 'cp' in user_lower:
            if 'from' in user_lower and 'to' in user_lower:
                from_idx = user_lower.find('from')
                to_idx = user_lower.find('to')
                
                if from_idx < to_idx:
                    source_words = user_words[user_lower[:from_idx].count(' '):user_lower[:to_idx].count(' ')]
                    source = self._extract_file_or_directory_name(source_words, user_input)
                    
                    dest_words = user_words[user_lower[:to_idx].count(' ') + 1:]
                    dest = self._extract_file_or_directory_name(dest_words, user_input, is_destination=True)
                    
                    if source and dest:
                        return self._build_copy_command(source, dest, user_lower, matched_output)
            
            elif 'to' in user_lower:
                to_idx = user_lower.find('to')
                copy_idx = user_lower.find('copy')
                
                if copy_idx < to_idx:
                    source_words = user_words[copy_idx + 1:user_lower[:to_idx].count(' ')]
                    filtered_source_words = [w for w in source_words if w.lower() not in ['files', 'all', 'directory', 'folder', 'dir']]
                    if not filtered_source_words:
                        filtered_source_words = source_words
                    
                    source = self._extract_file_or_directory_name(filtered_source_words, user_input)
                    
                    dest_words = user_words[user_lower[:to_idx].count(' ') + 1:]
                    dest = self._extract_file_or_directory_name(dest_words, user_input, is_destination=True)
                    
                if source and dest:
                    return self._build_copy_command(source, dest, user_lower, matched_output)
        
        descriptive_phrase_patterns = [
            r'using\s+a\s+\w+\s+(approach|way|method|manner|mode|style)',
            r'using\s+a\s+(simple|easy|quick|fast|basic|direct|straightforward)(\s+(approach|way|method|manner|mode|style))?',
            r'using\s+a\s+\w+$',
            r'in\s+a\s+\w+\s+(approach|way|method|manner|mode|style)',
            r'in\s+a\s+(simple|easy|quick|fast|basic|direct|straightforward)'
        ]
        has_descriptive_phrase = any(re.search(pattern, user_lower) for pattern in descriptive_phrase_patterns)
        
        dir_name = None
        if not has_descriptive_phrase:
            for i, word in enumerate(user_words):
                word_clean = word.strip("'\".,;:!?")
                word_lower_clean = word_clean.lower()
                
                if word_lower_clean in ['the', 'a', 'an', 'this', 'that', 'current', 'all', 'files', 'file', 'directory', 'folder', 'dir', 'in', 'to', 'from', 'using']:
                    continue
                
                descriptive_word_roots = ['simple', 'easy', 'quick', 'fast', 'basic', 'direct', 'straightforward']
                descriptive_phrases = ['approach', 'way', 'method', 'manner', 'mode', 'style']
                
                context_before = ' '.join(user_words[max(0, i-3):i]).lower() if i > 0 else ''
                context_after = ' '.join(user_words[i+1:min(len(user_words), i+3)]).lower() if i < len(user_words) - 1 else ''
                
                is_descriptive = False
                for root in descriptive_word_roots:
                    if (word_lower_clean == root or 
                        word_lower_clean.startswith(root) or 
                        root.startswith(word_lower_clean[:3])):
                        is_descriptive = True
                        break
                
                if is_descriptive:
                    if any(phrase in context_after for phrase in descriptive_phrases):
                        continue
                    if 'using a' in context_before or 'in a' in context_before:
                        continue
                
                if word_lower_clean in descriptive_phrases:
                    continue
                
                if (any(marker in context_before for marker in ['in', 'to', 'into', 'directory', 'folder']) or
                    any(marker in context_after for marker in ['directory', 'folder', 'dir']) or
                    (i > 0 and user_words[i-1].lower() == 'in') or
                    (i < len(user_words) - 1 and user_words[i+1].lower() in ['directory', 'folder', 'dir'])):
                    
                    is_descriptive_word = False
                    for root in descriptive_word_roots:
                        if (word_lower_clean == root or 
                            word_lower_clean.startswith(root) or 
                            root.startswith(word_lower_clean[:3])):
                            is_descriptive_word = True
                            break
                    
                    if not ('using a' in context_before and is_descriptive_word):
                        if word_clean and len(word_clean) > 1:
                            dir_name = word_clean
                            break
        
        if dir_name:
            if 'ls' in matched_output.lower():
                matched_parts = matched_output.split()
                flags = ''
                if len(matched_parts) > 1 and matched_parts[1].startswith('-'):
                    flags = ' ' + matched_parts[1]
                return f'ls{flags} {dir_name}'
            
            elif 'cd' in matched_output.lower():
                dir_lower = dir_name.lower()
                if dir_lower == 'parent':
                    return 'cd ..'
                elif dir_lower in ['home', '~']:
                    return 'cd ~'
                elif dir_lower in ['current', '.']:
                    return 'cd .'
                return f'cd {dir_name}'
        
        if 'show' in user_lower and 'current' in user_lower and 'directory' in user_lower:
            if 'pwd' in matched_output.lower():
                return 'pwd'
        
        if 'print' in user_lower and 'echo' in matched_output.lower():
            print_idx = user_lower.find('print')
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
        
        return matched_output
    
    def _extract_file_or_directory_name(self, words: list, original_input: str, is_destination: bool = False) -> Optional[str]:
        """Extract file or directory name from a list of words, handling special cases.
        Works for both files and directories.
        
        Takes in:
            words: List of words to extract from
            original_input: Original user input for context
            is_destination: Whether this is a destination (handles special paths)
        
        Gives back:
            File/directory name or special path (.., ~, .)"""
        
        if not words:
            return None
        
        for word in words:
            if word.startswith('/') or word.startswith('~') or word == '..' or word.startswith('../'):
                return word
            
            if word == '.' or word.strip() == '.':
                return '.'
            
            word_clean = word.strip("'\".,;:!?")
            word_lower = word_clean.lower()
            
            if word_lower == 'parent':
                return '..'
            elif word_lower in ['home', 'HOME']:
                return '~'
            elif word_lower in ['current']:
                return '.'
            
            if word_lower in ['the', 'a', 'an', 'this', 'that', 'directory', 'folder', 'dir', 'to', 'from', 'file', 'files']:
                continue
            
            if word_clean and len(word_clean) > 0:
                if '/' in word_clean:
                    return word_clean
                if (word_clean[0].isupper() or '_' in word_clean or 
                    (word_clean.replace('_', '').replace('.', '').isalnum() and word_lower not in ['all', 'files', 'file'])):
                    return word_clean
        
        if words:
            last_word = words[-1]
            if last_word.startswith('/') or last_word.startswith('~') or last_word == '..' or last_word.startswith('../'):
                return last_word
            
            if last_word == '.' or last_word.strip() == '.':
                return '.'
            
            last_word_clean = last_word.strip("'\".,;:!?")
            if last_word_clean and len(last_word_clean) > 0 and last_word_clean.replace('_', '').replace('.', '').isalnum():
                return last_word_clean
        
        return None
    
    def _handle_file_operations(self, user_input: str) -> Optional[str]:
        """Handle file operations (create, write, append, read) before ML matching.
        
        Takes in:
            user_input: Natural language prompt
        
        Gives back:
            Command string if file operation detected, None otherwise"""
        
        user_lower = user_input.lower()
        user_words = user_input.split()
        
        if 'create' in user_lower and 'file' in user_lower:
            file_name = self._extract_file_name_from_create(user_input, user_lower, user_words)
            if file_name:
                return f'touch {file_name}'
        
        if 'write' in user_lower and 'to' in user_lower:
            content, file_name = self._extract_write_operation(user_input, user_lower, user_words)
            if content and file_name:
                if content.startswith("'") and content.endswith("'"):
                    return f"echo {content} > {file_name}"
                elif content.startswith('"') and content.endswith('"'):
                    return f'echo {content} > {file_name}'
                else:
                    return f"echo '{content}' > {file_name}"
        
        if ('add' in user_lower or 'append' in user_lower) and 'to' in user_lower:
            content, file_name = self._extract_append_operation(user_input, user_lower, user_words)
            if content and file_name:
                if content.startswith("'") and content.endswith("'"):
                    return f"echo {content} >> {file_name}"
                elif content.startswith('"') and content.endswith('"'):
                    return f'echo {content} >> {file_name}'
                else:
                    return f"echo '{content}' >> {file_name}"
        
        if 'read' in user_lower and 'file' in user_lower:
            file_name = self._extract_file_name_from_read(user_input, user_lower, user_words)
            if file_name:
                return f'cat {file_name}'
        
        return None
    
    def _extract_file_name_from_create(self, user_input: str, user_lower: str, user_words: list) -> Optional[str]:
        """Extract file name from create file commands.
        
        Takes in:
            user_input: Original user input
            user_lower: Lowercase user input
            user_words: List of words from user input
        
        Gives back:
            File name with extension preserved"""
        
        named_idx = user_lower.find('named')
        called_idx = user_lower.find('called')
        file_idx = user_lower.find('file')
        
        if named_idx != -1:
            start_idx = named_idx + len('named')
            remaining = user_input[start_idx:].strip()
            words = remaining.split()
            return self._extract_file_name_from_words(words, user_input)
        
        if called_idx != -1:
            start_idx = called_idx + len('called')
            remaining = user_input[start_idx:].strip()
            words = remaining.split()
            return self._extract_file_name_from_words(words, user_input)
        
        if file_idx != -1:
            for i in range(file_idx + 1, len(user_words)):
                word = user_words[i]
                word_clean = word.strip("'\".,;:!?")
                if word_clean.lower() in ['named', 'called', 'the', 'a', 'an']:
                    continue
                if word_clean and ('.' in word_clean or word_clean[0].isupper() or '_' in word_clean):
                    return self._extract_file_name_from_words(user_words[i:], user_input)
        
        return None
    
    def _extract_file_name_from_words(self, words: list, original_input: str) -> Optional[str]:
        """Extract file name from a list of words, preserving extension and quotes.
        
        Takes in:
            words: List of words to extract from
            original_input: Original user input for context
        
        Gives back:
            File name with extension and quotes preserved if needed"""
        
        if not words:
            return None
        
        file_parts = []
        for word in words:
            word_clean = word.strip("'\".,;:!?")
            word_lower = word_clean.lower()
            
            if word_lower in ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'with', 'and', 'or']:
                if file_parts:
                    break
                continue
            
            if word.startswith('/') or word.startswith('~'):
                return word
            
            if word_clean:
                if '.' in word_clean:
                    file_parts.append(word)
                    break
                elif word_clean[0].isupper() or '_' in word_clean:
                    file_parts.append(word)
                elif word_clean.replace('.', '').replace('_', '').isalnum() and len(word_clean) > 1:
                    file_parts.append(word)
                    if '.' in word:
                        break
        
        if file_parts:
            result = ' '.join(file_parts)
            if ' ' in result and not (result.startswith("'") and result.endswith("'")):
                if not (result.startswith('"') and result.endswith('"')):
                    result = f"'{result}'"
            return result
        
        if words:
            first_word = words[0].strip("'\".,;:!?")
            if first_word and ('.' in first_word or first_word.replace('.', '').replace('_', '').isalnum()):
                return words[0]
        
        return None
    
    def _extract_write_operation(self, user_input: str, user_lower: str, user_words: list) -> Tuple[Optional[str], Optional[str]]:
        """Extract content and file name from write operations.
        
        Takes in:
            user_input: Original user input
            user_lower: Lowercase user input
            user_words: List of words from user input
        
        Gives back:
            Tuple of (content, file_name)"""
        
        write_idx = user_lower.find('write')
        to_idx = user_lower.find('to')
        file_idx = user_lower.find('file')
        
        if write_idx == -1 or to_idx == -1:
            return None, None
        
        if write_idx < to_idx:
            to_word_idx = user_lower[:to_idx].count(' ')
            content_words = user_words[write_idx + 1:to_word_idx]
            content = ' '.join(content_words).strip()
            
            file_words = user_words[to_word_idx + 1:]
            
            if file_idx != -1 and file_idx > to_idx:
                file_word_idx = user_lower[:file_idx].count(' ')
                if file_word_idx >= to_word_idx:
                    file_words = user_words[file_word_idx:]
            
            file_name = self._extract_file_name_from_words(file_words, user_input)
            
            if not file_name and file_words:
                first_word = file_words[0].strip("'\".,;:!?")
                if '.' in first_word or first_word.replace('.', '').replace('_', '').isalnum():
                    file_name = file_words[0]
            
            if content and file_name:
                return content, file_name
        
        return None, None
    
    def _extract_append_operation(self, user_input: str, user_lower: str, user_words: list) -> Tuple[Optional[str], Optional[str]]:
        """Extract content and file name from append/add operations.
        
        Takes in:
            user_input: Original user input
            user_lower: Lowercase user input
            user_words: List of words from user input
        
        Gives back:
            Tuple of (content, file_name)"""
        
        add_idx = user_lower.find('add')
        append_idx = user_lower.find('append')
        to_idx = user_lower.find('to')
        file_idx = user_lower.find('file')
        
        action_idx = add_idx if add_idx != -1 else append_idx
        if action_idx == -1 or to_idx == -1:
            return None, None
        
        if action_idx < to_idx:
            to_word_idx = user_lower[:to_idx].count(' ')
            content_words = user_words[action_idx + 1:to_word_idx]
            content = ' '.join(content_words).strip()
            
            file_words = user_words[to_word_idx + 1:]
            
            if file_idx != -1 and file_idx > to_idx:
                file_word_idx = user_lower[:file_idx].count(' ')
                if file_word_idx >= to_word_idx:
                    file_words = user_words[file_word_idx:]
            
            file_name = self._extract_file_name_from_words(file_words, user_input)
            
            if not file_name and file_words:
                first_word = file_words[0].strip("'\".,;:!?")
                if '.' in first_word or first_word.replace('.', '').replace('_', '').isalnum():
                    file_name = file_words[0]
            
            if content and file_name:
                return content, file_name
        
        return None, None
    
    def _extract_file_name_from_read(self, user_input: str, user_lower: str, user_words: list) -> Optional[str]:
        """Extract file name from read file commands.
        
        Takes in:
            user_input: Original user input
            user_lower: Lowercase user input
            user_words: List of words from user input
        
        Gives back:
            File name with extension preserved"""
        
        read_idx = user_lower.find('read')
        file_idx = user_lower.find('file')
        
        if read_idx == -1 or file_idx == -1:
            return None
        
        if read_idx < file_idx:
            file_word_idx = user_lower[:file_idx].count(' ')
            file_words = user_words[file_word_idx + 1:]
            if not file_words:
                file_words = user_words[file_word_idx:]
            
            filtered_file_words = [w for w in file_words if w.lower() not in ['file', 'the', 'a', 'an']]
            if not filtered_file_words:
                filtered_file_words = file_words
            
            file_name = self._extract_file_name_from_words(filtered_file_words, user_input)
            
            if not file_name and filtered_file_words:
                first_word = filtered_file_words[0].strip("'\".,;:!?")
                if '.' in first_word or first_word.replace('.', '').replace('_', '').isalnum():
                    file_name = filtered_file_words[0]
            
            return file_name
        
        return None
    
    def _build_copy_command(self, source: str, dest: str, user_lower: str, matched_output: str) -> str:
        """Build a cp command from source and destination.
        
        Takes in:
            source: Source file/directory
            dest: Destination path
            user_lower: Lowercase user input for context
            matched_output: Matched output from dataset
        
        Gives back:
            cp command string"""
        
        dest_lower = dest.lower()
        if dest_lower == 'parent':
            dest = '..'
        elif dest == '..' or dest.startswith('../'):
            dest = dest
        elif dest_lower in ['home', '~', 'home']:
            dest = '~'
        elif dest_lower in ['current', '.']:
            dest = '.'
        
        source_lower = source.lower()
        if source_lower in ['home', 'HOME']:
            source = '~'
        
        
        is_file = False
        is_directory = False
        
        if 'directory' in user_lower or 'folder' in user_lower:
            source_pos = user_lower.find(source.lower())
            dir_pos = user_lower.find('directory')
            folder_pos = user_lower.find('folder')
            if (dir_pos != -1 and abs(dir_pos - source_pos) < 10) or (folder_pos != -1 and abs(folder_pos - source_pos) < 10):
                is_directory = True
        
        if '.' in source and '/' not in source:
            parts = source.split('.')
            if len(parts) > 1:
                extension = parts[-1].lower()
                common_extensions = ['txt', 'pdf', 'jpg', 'jpeg', 'png', 'gif', 'csv', 'json', 'xml', 
                                   'py', 'js', 'html', 'css', 'md', 'sh', 'zip', 'tar', 'gz', 'log']
                if extension in common_extensions and len(extension) <= 5:
                    is_file = True
        
        should_add_wildcard = False
        
        if ('all' in user_lower and 'files' in user_lower):
            if not is_file:
                should_add_wildcard = True
        elif 'files' in user_lower and 'from' in user_lower:
            if 'files' in user_lower and user_lower.count('files') >= user_lower.count('file'):
                if not is_file:
                    should_add_wildcard = True
        elif '/*' in matched_output and not is_file and ('directory' in user_lower or 'folder' in user_lower):
            should_add_wildcard = True
        
        if should_add_wildcard:
            if not ('*' in source or '/' in source or source.endswith('.*')):
                return f'cp {source}/* {dest}'
        
        needs_recursive = False
        if is_directory:
            needs_recursive = True
        elif not is_file and not should_add_wildcard:
            needs_recursive = True
        
        if needs_recursive:
            return f'cp -r {source} {dest}'
        else:
            return f'cp {source} {dest}'
    
    def _extract_directory_name(self, words: list, original_input: str, is_destination: bool = False) -> Optional[str]:
        """Extract directory name from a list of words, handling special cases.
        
        Takes in:
            words: List of words to extract from
            original_input: Original user input for context
            is_destination: Whether this is a destination (handles special paths)
        
        Gives back:
            Directory name or special path (.., ~, .)"""
        
        if not words:
            return None
        
        for word in words:
            if word == '..' or word.startswith('../'):
                return word
            
            word_clean = word.strip("'\".,;:!?")
            word_lower = word_clean.lower()
            
            if word_lower == 'parent':
                return '..'
            elif word_lower in ['home', '~']:
                return '~'
            elif word_lower in ['current']:
                return '.'
            elif word_clean == '.':
                return '.'
            
            if word_lower in ['the', 'a', 'an', 'this', 'that', 'directory', 'folder', 'dir', 'to', 'from']:
                continue
            
            if word_clean and len(word_clean) > 1:
                if (word_clean[0].isupper() or '_' in word_clean or 
                    (word_clean.replace('_', '').isalnum() and word_lower not in ['all', 'files', 'file'])):
                    return word_clean
        
        if words:
            last_word = words[-1]
            if last_word == '..' or last_word.startswith('../'):
                return last_word
            
            last_word_clean = last_word.strip("'\".,;:!?")
            if last_word_clean and len(last_word_clean) > 1 and last_word_clean.replace('_', '').isalnum():
                return last_word_clean
        
        return None
    
    def _preprocess_text(self, text: str, normalize_entities: bool = True) -> str:
        """Preprocess text for pattern learning using NLP pipeline.
        Normalizes directory/file names to learn patterns like "list files in [DIR]" 
        regardless of the actual directory name.
        
        Takes in:
        text: Input text
        normalize_entities: If True, normalize directory names to learn patterns
        Gives back:
        preprocessed text optimized for TF-IDF pattern matching"""

        if normalize_entities:
            words = text.split()
            normalized_words = []
            i = 0
            
            text_lower = text.lower()
            descriptive_phrase_patterns = [
                r'using\s+a\s+\w+\s+(approach|way|method|manner|mode|style)',
                r'using\s+a\s+(simple|easy|quick|fast|basic|direct|straightforward)(\s+(approach|way|method|manner|mode|style))?',
                r'using\s+a\s+\w+$',
            ]
            has_descriptive_phrase = any(re.search(pattern, text_lower) for pattern in descriptive_phrase_patterns)
            
            while i < len(words):
                word = words[i]
                word_clean = word.strip("'\".,;:!?")
                word_lower = word_clean.lower()
                
                if has_descriptive_phrase:
                    descriptive_word_roots = ['simple', 'easy', 'quick', 'fast', 'basic', 'direct', 'straightforward']
                    descriptive_phrases = ['approach', 'way', 'method', 'manner', 'mode', 'style']
                    context_before = ' '.join(words[max(0, i-3):i]).lower() if i > 0 else ''
                    
                    is_descriptive = False
                    for root in descriptive_word_roots:
                        if (word_lower == root or 
                            word_lower.startswith(root) or 
                            root.startswith(word_lower[:3])):
                            is_descriptive = True
                            break
                    
                    if is_descriptive and 'using a' in context_before:
                        i += 1
                        continue
                    if word_lower in descriptive_phrases:
                        i += 1
                        continue
                
                context_before = ' '.join(words[max(0, i-2):i]).lower() if i > 0 else ''
                context_after = ' '.join(words[i+1:min(len(words), i+3)]).lower() if i < len(words) - 1 else ''
                
                if (any(marker in context_before for marker in ['in', 'to', 'into', 'directory', 'folder']) or
                    any(marker in context_after for marker in ['directory', 'folder', 'dir']) or
                    (i > 0 and words[i-1].lower() == 'in') or
                    (i < len(words) - 1 and words[i+1].lower() in ['directory', 'folder', 'dir'])):
                    
                    if word_clean and len(word_clean) > 1:
                        if (word_clean[0].isupper() or '_' in word_clean or 
                            (word_clean.replace('_', '').isalnum() and word_lower not in ['the', 'a', 'an', 'current', 'this', 'that', 'all', 'files', 'file', 'simple', 'easy', 'quick', 'fast', 'basic', 'direct', 'straightforward'])):
                            normalized_words.append('DIRNAME')
                            if i < len(words) - 1 and words[i+1].lower() in ['directory', 'folder', 'dir']:
                                i += 2
                                continue
                            else:
                                i += 1
                                continue
                
                normalized_words.append(word)
                i += 1
            
            text = ' '.join(normalized_words)

        keywords = self.nlp_pipeline.get_keywords(text, filter_stopwords=True)
        

        result = self.nlp_pipeline.process(text)
        doc = result['Doc']
        


        important_lemmas = []
        verbs = []
        nouns = []
        
        for token in doc:


            if token.pos_ == 'VERB':
                lemma = token.lemma_.lower()
                important_lemmas.append(lemma)
                verbs.append(lemma)

                if lemma not in ['be', 'have', 'do']:
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
        
        if 'direct' in processed_text and 'directory' not in processed_text:
            processed_text += ' directory'
        
        return processed_text if processed_text else text.lower()
    
    def _build_model(self):
        """Builds the TF-IDF model for pattern learning."""
        print("Building TF-IDF model...")
        
        processed_inputs = [self._preprocess_text(text, normalize_entities=True) for text in self.input_texts]
        


        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        action_verbs_to_keep = {'Go', 'Do', 'Make', 'Get', 'Set', 'Run', 'Show', 'List', 'Find', 'Copy', 'Move', 'Delete', 'Remove', 'Create', 'Change', 'Enter', 'Back', 'Previous'}
        custom_stop_words = list(ENGLISH_STOP_WORDS - action_verbs_to_keep)
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            stop_words=custom_stop_words,
            analyzer='word',
            lowercase=True,
            norm='l2'
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
        
        file_operation = self._handle_file_operations(user_input)
        if file_operation:
            return file_operation
        
        short_command_match = self._handle_short_commands(user_input)
        if short_command_match:
            return short_command_match
        
        best_match = self._find_best_match(user_input, top_k)
        if best_match:
            return best_match
        
        rule_match = self._rule_based_match(user_input)
        if rule_match:
            return rule_match
        
        return f"echo 'Command not found for: {user_input}'"
    
    def _translate_single_step(self, user_input: str, top_k: int = 3) -> str:
        """Translate a single step (used by multi-step handler to avoid recursion).
        Follows the same pipeline as translate() but skips multi-step detection.
        
        Takes in:
            user_input: Natural language prompt for a single step
            top_k: Number of top matches to consider
        
        Gives back:
            Linux command or error message"""
        
        if self.malware_detector.is_malware(user_input):
            return "echo 'Malware detected'"
        
        if self._is_direct_command(user_input):
            return user_input
        
        short_command_match = self._handle_short_commands(user_input)
        if short_command_match:
            return short_command_match
        
        best_match = self._find_best_match(user_input, top_k)
        if best_match:
            return best_match
        
        rule_match = self._rule_based_match(user_input)
        if rule_match:
            return rule_match
        
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
        
        if len(text.split()) == 1:
            common_commands = ['ls', 'pwd', 'cd', 'cat', 'grep', 'find', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 
                             'echo', 'export', 'unset', 'env', 'which', 'clear', 'exit', 'quit', 'help',
                             'head', 'tail', 'less', 'more', 'wc', 'sort', 'uniq', 'cut', 'awk', 'sed']
            if text_lower in common_commands:
                return True
        
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
            r'^[a-z]+\s+[-/]',
            r'^[a-z]+\s+/',
            r'^[a-z]+\s+\./',
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
        Only triggers on explicit sequential markers, not single commands.
        
        Takes in:
        text: Input text
        Gives back:
        true if multi-step detected"""

        text_lower = text.lower()
        
        sequential_markers = [
            ' then ', ' and then ', ' after that ', ' after ', ' next ',
            ' followed by ', ' subsequently ', ' afterwards '
        ]
        
        for marker in sequential_markers:
            if marker in text_lower:
                return True
        
        if ' and ' in text_lower:
            words = text_lower.split()
            and_positions = [i for i, w in enumerate(words) if w == 'and']
            
            for and_idx in and_positions:
                left_side = ' '.join(words[:and_idx])
                right_side = ' '.join(words[and_idx+1:])
                
                single_action_patterns = [
                    'files in', 'directory', 'folder', 'in the', 'in a'
                ]
                
                if any(pattern in text_lower for pattern in single_action_patterns):
                    return False
                
                action_verbs = ['list', 'show', 'display', 'print', 'copy', 'move', 'delete', 'create', 'go', 'enter', 'change', 'cd', 'ls']
                left_has_action = any(verb in left_side for verb in action_verbs)
                right_has_action = any(verb in right_side for verb in action_verbs)
                
                if left_has_action and right_has_action and left_side != right_side:
                    if len(left_side.split()) > 2 and len(right_side.split()) > 2:
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
            
            exact_match_found = False
            for idx, training_input in enumerate(self.input_texts):
                if training_input.lower().strip() == step_lower:
                    final_steps.append(step)
                    exact_match_found = True
                    break
            
            if exact_match_found:
                continue


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
                command = self._translate_single_step(step, top_k=5)
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
            word_original = word.strip("'\".,;:!?")
            word_lower = word_original.lower()

            if word_lower in ['the', 'a', 'an', 'this', 'that', 'all', 'files', 'file', 'directory', 'folder', 'dir', 'in', 'to', 'from']:
                continue

            if '/' in word_original:
                directory_names.append(word_original)

            elif '_' in word_original and word_lower not in common_words:
                context_start = max(0, i-3)
                context_end = min(len(words), i+3)
                context = ' '.join(words[context_start:context_end]).lower()
                if any(marker in context for marker in ['to', 'into', 'directory', 'folder', 'change', 'enter', 'go', 'the', 'in the', 'in']):
                    directory_names.append(word_original)

            elif word_original.replace('_', '').isalnum() and len(word_original) > 2:
                context_start = max(0, i-3)
                context_end = min(len(words), i+3)
                context = ' '.join(words[context_start:context_end]).lower()
                
                if i > 0 and words[i-1].lower() in ['folder', 'directory', 'dir', 'in']:
                    if words[i-1].lower() == 'in' or (i < len(words) - 1 and words[i+1].lower() in ['folder', 'directory', 'dir']):
                        directory_names.append(word_original)

                elif i < len(words) - 1 and words[i+1].lower() in ['folder', 'directory', 'dir']:
                    directory_names.append(word_original)
                
                elif i > 1 and words[i-2].lower() == 'in' and words[i-1].lower() == 'the' and i < len(words) - 1 and words[i+1].lower() in ['directory', 'folder', 'dir']:
                    directory_names.append(word_original)

                elif any(marker in context for marker in ['to', 'into', 'change', 'enter', 'go']) and word_lower not in ['files', 'file', 'items', 'content', 'list', 'show', 'display', 'print']:
                    if word_original[0].isupper() or word_original.isupper():
                        directory_names.append(word_original)
                    elif any(marker in context for marker in ['directory', 'folder']):
                        directory_names.append(word_original)
        

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
        

        if any(word in text_lower for word in ['List', 'Show', 'Display', 'Print']) and any(word in text_lower for word in ['File', 'Files', 'Content', 'Items']):
            if not directory_names:
                if 'detail' in text_lower or 'all' in text_lower or 'hidden' in text_lower:
                    return 'ls -la'
                elif 'recursive' in text_lower or 'subdirectory' in text_lower:
                    return 'ls -laR'
                else:
                    return 'ls'
        
        if any(word in text_lower for word in ['List', 'Show', 'Display', 'Print']) and any(word in text_lower for word in ['In', 'Folder', 'Directory']):
            if directory_names:
                dir_name = directory_names[0]
                if 'detail' in text_lower or 'hidden' in text_lower:
                    return f'ls -la {dir_name}'
                else:
                    return f'ls {dir_name}'
        


        cd_patterns = [
            'change directory', 'go to', 'navigate to', 'switch to', 'move to', 
            'enter directory', 'enter the directory', 'cd'
        ]

        starts_with_enter = text_lower.startswith('enter ')
        
        if any(pattern in text_lower for pattern in cd_patterns) or \
           (starts_with_enter and (directory_names or 'directory' in text_lower)):
            if directory_names:
                dir_name = directory_names[0]
                return f'cd {dir_name}'
            elif starts_with_enter and not directory_names:

                words_after_enter = user_input.split()[1:]
                if words_after_enter:

                    potential_dir = words_after_enter[0]

                    if potential_dir.lower() not in ['the', 'a', 'an', 'directory', 'folder', 'dir']:
                        return f'cd {potential_dir}'
            elif 'home' in text_lower:
                return 'cd ~'
            elif 'root' in text_lower:
                return 'cd /'
            elif 'previous' in text_lower or 'back' in text_lower:
                return 'cd -'
            elif 'up' in text_lower or 'parent' in text_lower:
                return 'cd ..'
            else:
                return 'cd'
        
        if text_lower.strip() == 'go back' or text_lower.strip() == 'go to previous':
            return 'cd -'
        


        if any(word in text_lower for word in ['current directory', 'working directory', 'where am i', 'pwd', 'show directory']):
            return 'pwd'

        if 'show' in text_lower and ('current' in text_lower or 'working' in text_lower) and ('directory' in text_lower or 'folder' in text_lower):
            return 'pwd'
        
        if ('show' in text_lower or 'display' in text_lower) and 'current' in text_lower and ('directory' in text_lower or 'direct' in text_lower or 'folder' in text_lower):
            return 'pwd'
        
        if 'print' in text_lower and 'current' in text_lower and ('directory' in text_lower or 'direct' in text_lower or 'folder' in text_lower):
            return 'pwd'
        
        if 'show' in text_lower and 'me' in text_lower and 'current' in text_lower and ('directory' in text_lower or 'direct' in text_lower):
            return 'pwd'
        

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
        

        processed_input = self._preprocess_text(user_input, normalize_entities=True)
        

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
        if word_count <= 2:
            threshold = 0.05
        elif word_count <= 3:
            threshold = 0.08
        else:
            threshold = 0.12
        
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
            if not word_clean:
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

            elif word_clean.isalnum() and len(word_clean) > 2 and '_' not in word_clean:
                if word_clean[0].isupper() or is_directory:
                    if is_directory:
                        directory_names.append(word_clean)
                    elif is_file:
                        file_names.append(word_clean)
                elif word_clean.isalnum() and not word_clean[0].isupper():
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

            if 'previous' in user_lower or ('back' in user_lower and ('go' in user_lower or 'navigate' in user_lower)):
                if 'previous directory' in user_lower or 'go back' in user_lower or 'navigate back' in user_lower or 'change to previous' in user_lower:
                    return 'cd -'
            if '~' in user_input or user_input.strip() == 'Go to ~':
                return 'cd ~'
            if 'home' in user_lower and ('directory' in user_lower or 'go to' in user_lower or 'navigate' in user_lower or user_input.strip() == 'Go to home'):
                return 'cd ~'
            if '.' in user_input and ('to .' in user_lower or 'to current' in user_lower):
                return 'cd .'
        


        if directory_names:
            user_dir = directory_names[0]
            

            if user_dir.upper() == 'HOME' and 'Home' in user_lower:
                user_dir = '~'
            elif user_dir.lower() == 'Current' or ('.' in user_input and user_dir == '.'):
                user_dir = '.'
            
            if command.startswith('cd ') or command.lower().startswith('cd '):

                return f'cd {user_dir}'
            elif command.startswith('ls') or command.lower().startswith('ls'):

                cmd_parts = command.split()
                if len(cmd_parts) == 1:
                    return f'ls {user_dir}'
                elif len(cmd_parts) >= 2:
                    return f'ls {user_dir}'
            elif command.startswith('cp ') or command.lower().startswith('cp '):

                if 'from' in user_lower and 'to' in user_lower:
                    from_idx = user_lower.find('from')
                    to_idx = user_lower.find('to')
                    if from_idx < to_idx:
                        source_words = user_input.split()[user_lower[:from_idx].count(' '):user_lower[:to_idx].count(' ')]
                        source = self._extract_directory_name(source_words, user_input)
                        
                        dest_words = user_input.split()[user_lower[:to_idx].count(' ') + 1:]
                        dest = self._extract_directory_name(dest_words, user_input, is_destination=True)
                        
                        if source and dest:
                            dest_lower = dest.lower()
                            if dest_lower == 'parent':
                                dest = '..'
                            elif dest == '..' or dest.startswith('../'):
                                dest = dest
                            elif dest_lower in ['home', '~']:
                                dest = '~'
                            elif dest_lower in ['current', '.']:
                                dest = '.'
                            
                            if '/*' in command or 'all' in user_lower or 'files' in user_lower:
                                return f'cp {source}/* {dest}'
                            else:
                                return f'cp {source} {dest}'
        

        if similarity > 0.8:
            return command
        

        if similarity >= 0.3 and similarity <= 0.7:

            if (command.startswith('cd ') or command.lower().startswith('cd ')) and directory_names:

                return f'cd {directory_names[0]}'
            elif (command.startswith('ls') or command.lower().startswith('ls')) and directory_names:

                cmd_parts = command.split()
                if len(cmd_parts) == 1 or (len(cmd_parts) == 2 and cmd_parts[1] not in directory_names):
                    return f'ls {directory_names[0]}'
            elif (command.startswith('cp ') or command.lower().startswith('cp ')) and directory_names:

                user_lower = user_input.lower()
                if 'From' in user_lower and 'To' in user_lower:

                    from_idx = user_lower.find('From')
                    to_idx = user_lower.find('To')
                    if from_idx < to_idx:
                        source = directory_names[0] if directory_names else 'Source'
                        dest = '.' if '.' in user_input else (directory_names[1] if len(directory_names) > 1 else '.')
                        return f'cp {source}/* {dest}'
        
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
        "Delete all files",
    ]
    
    print("\nTesting Command Agent")
    print("=" * 60)
    
    for test_input in test_cases:
        print(f"\nInput: {test_input}")
        output = agent.translate(test_input)
        print(f"Output: {output}")

