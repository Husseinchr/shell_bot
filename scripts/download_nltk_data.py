"""Script to download all required NLTK data for the NLP project."""

import nltk
import sys

def download_nltk_data():
    """Download all required NLTK data packages."""
    print("Downloading NLTK data packages...")
    
    packages = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4'),
        ('stopwords', 'corpora/stopwords'),
    ]
    
    for package_name, data_path in packages:
        try:
            nltk.data.find(data_path)
            print(f"✓ {package_name} already downloaded")
        except LookupError:
            print(f"Downloading {package_name}...")
            try:
                nltk.download(package_name, quiet=False)
                print(f"✓ {package_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {package_name}: {e}")
                return False
    
    print("\nAll NLTK data packages are ready!")
    return True

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)

