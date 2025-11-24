# NLP Command Agent - Natural Language to Linux Commands

An AI-powered agent that translates natural language prompts into Linux commands using a 5-step NLP pipeline. The agent is integrated into a Python shell (`src/mini_shell.py`) to allow users to type natural language instead of commands.

## Features

- **5-Step NLP Pipeline**: Implements tokenization, lemmatization, POS tagging, dependency parsing, and word meaning extraction
- **Natural Language Translation**: Converts prompts like "I want to list all files" to `ls -la`
- **Multi-Step Commands**: Handles complex requests like "enter directory then list files then copy file"
- **Malware Detection**: Detects and blocks dangerous commands before execution
- **Shell Integration**: Seamlessly integrated into `src/mini_shell.py`

## Installation

### Option 1: Using Virtual Environment (Recommended)

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Install Python dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

3. Download required NLP models:
```bash
# Download spaCy English model
python3 -m spacy download en_core_web_sm

# Download NLTK data (will be downloaded automatically on first run, or run manually):
python3 scripts/download_nltk_data.py
# OR manually:
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

### Option 2: Global Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLP models:
```bash
# Download spaCy English model
python3 -m spacy download en_core_web_sm

# Download NLTK data (will be downloaded automatically on first run, or run manually):
python3 scripts/download_nltk_data.py
# OR manually:
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

## Usage

### Running the Shell

**Option 1: Using the run script (easiest)**
```bash
./run_shell.sh
```

**Option 2: Run directly**
```bash
source venv/bin/activate
python3 src/mini_shell.py
```

**Option 3: Run as module**
```bash
source venv/bin/activate
python3 -m src.mini_shell
```

**Option 4: Without virtual environment (if installed globally)**
```bash
python3 src/mini_shell.py
```

### Using Natural Language Commands

Once the shell is running, you can type natural language instead of commands:

```
$ I want to list all files in this directory
→ ls -la
[file listing output]

$ Show me the current directory
→ pwd
/home/user/project

$ I want to enter the directory data then list the files in it
→ cd data; ls -la
[file listing output]
```

### Direct Commands Still Work

You can still use direct commands as usual:

```
$ ls -la
[file listing output]

$ cd /tmp
$ pwd
/tmp
```

## Architecture

### Components

1. **src/nlp_pipeline.py**: Implements the 5-step NLP processing
   - Step 1: Tokenization/Segmentation (NLTK)
   - Step 2: Lemmatization (spaCy)
   - Step 3: POS Tagging (spaCy)
   - Step 4: Dependency Parsing/Chunking (spaCy)
   - Step 5: Word Meaning (WordNet)

2. **src/malware_detector.py**: Detects dangerous commands using:
   - Pattern matching for known dangerous patterns
   - Semantic analysis using WordNet
   - Dependency parsing to identify destructive actions

3. **src/command_agent.py**: Main agent that:
   - Loads training data from `Dataset/linuxcommands.json`
   - Uses TF-IDF for semantic matching
   - Handles multi-step command translation
   - Integrates with malware detector

4. **src/mini_shell.py**: Modified shell with natural language support

### Training Data

The agent is trained on `Dataset/linuxcommands.json` which contains 8,669 examples of natural language to command mappings.

## Testing

Run the test suite:

```bash
python3 scripts/test_translations.py
```

This will test:
- NLP pipeline (all 5 steps)
- Malware detection
- Command translation
- Multi-step commands
- Shell integration

## Examples

### Simple Commands
```
Input: "I want to list all files in this directory"
Output: ls -la

Input: "Show me the current directory"
Output: pwd

Input: "Compress the file data.txt using bzip2"
Output: bzip2 data.txt
```

### Multi-Step Commands
```
Input: "I want to enter the directory data then list the files in it"
Output: cd data; ls -la

Input: "I want to enter the directory data then list the files in it and in every subdirectory of it then copy a specific file to another directory called temp"
Output: cd data; find . -type f; cp file.txt temp/
```

### Malware Detection
```
Input: "Delete all files in /etc directory"
Output: echo 'malware detected'

Input: "rm -rf /"
Output: echo 'malware detected'
```

## Implementation Details

### NLP Pipeline Steps

1. **Tokenization**: Uses NLTK's `word_tokenize` and `sent_tokenize` to break text into tokens and sentences
2. **Lemmatization**: Uses spaCy's built-in lemmatization to normalize words to their base forms
3. **POS Tagging**: Uses spaCy's POS tagger to identify parts of speech
4. **Dependency Parsing**: Uses spaCy's dependency parser to extract grammatical relations and noun chunks
5. **Word Meaning**: Uses NLTK WordNet for semantic similarity and meaning extraction

### Command Matching

The agent uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to match user input to training examples. It:
- Preprocesses text using the NLP pipeline to extract keywords
- Vectorizes both training data and user input
- Calculates cosine similarity to find the best match
- Returns the command from the most similar training example

### Malware Detection

The malware detector uses multiple strategies:
- Pattern matching for known dangerous command patterns
- Semantic analysis to detect destructive intent
- Dependency parsing to identify dangerous action-object relationships
- System directory targeting detection

## Limitations

- The agent works best with commands present in the training dataset
- Complex or very specific requests may not translate perfectly
- Malware detection may have false positives/negatives
- Performance depends on the size of the training dataset

## Future Improvements

- Fine-tune malware detection to reduce false positives
- Add support for more complex command combinations
- Improve multi-step command parsing
- Add command history and learning from user corrections
- Support for more natural language variations

## License

This project is part of an NLP course assignment.

## Author

CCE Student - NLP Course Project

