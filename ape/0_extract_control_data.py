import argparse
import random
import re
import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up input argument parser
def get_args():
    parser = argparse.ArgumentParser(description="Extract random 'neutral', control sentences from Wikipedia in a given language.")
    parser.add_argument("--lang", type=str, default="en", help="Language code (en, zh, ru, hi, mt, fr, nl, tl)")
    parser.add_argument("--num", type=int, default=1000, help="Number of sentences to extract")
    parser.add_argument("--out", type=str, default=None, help="Output file name")
    return parser.parse_args()

def extract_sentences(lang_code, num_sentences, output_file):
    # Set default file name as langcode_control.txt
    if not output_file:
        output_file = f"{lang_code}_control.txt"

    print(f"Initializing {lang_code} Wikipedia stream...")

    # Sent extraction configuration based on language characteristics
    # min_len sets minimum length of a sentence to be extracted (depending on typical sent length for a lang)
    # use_words: if True, splits sentence by space then counts number of words (for word-based languages, e.g. en, fr)
    #            if False, counts string length (for character-based languages, e.g. zh)
    # split_regex: punctuations to help regex delineate sentences for zh and hi
    # punct: valid punctuations in the language; used to check if a tokenized chunk is a valid sent or not
    config = {
        "zh": {"min_len": 50, "use_words": False, "split_regex": r'(?<=[。！？.!?])', "punct": ('。', '？', '！', '.', '?', '!')},
        "hi": {"min_len": 25, "use_words": True, "split_regex": r'(?<=[।।॥?!.])', "punct": ('।', '॥', '?', '!', '.')},
        "ru": {"min_len": 25, "use_words": True, "split_regex": None, "punct": ('.', '?', '!')},
        "default": {"min_len": 30, "use_words": True, "split_regex": None, "punct": ('.', '?', '!')}
    }

    # Set extraction configuration based on language
    set_cfg = config.get(lang_code, config["default"])
    
    # Load dataset based on language code
    # Set up fallback in case language has no data for latest dump
    try:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang_code}", split="train", streaming=True)
    except Exception:
        dataset = load_dataset("wikimedia/wikipedia", f"20220301.{lang_code}", split="train", streaming=True)

    # Set up list to contain extracted sentenes
    sentences = []
    # Randomly shuffle dataset
    shuffled_dataset = dataset.shuffle(buffer_size=10_000, seed=42)

    # Initialize TQDM progress bar
    pbar = tqdm(total=num_sentences, desc=f"Collecting {lang_code} sentences", unit="sent")

    # Iterate through shuffled dataset
    for entry in shuffled_dataset:
        # If there are already enough sentences extracted, exit loop
        if len(sentences) >= num_sentences:
            break

        # Get text from dataset entry
        text = entry['text']
        
        # Implement sentence splitting Logic
        # If language has a split_regex configuration, split sentences using that
        if set_cfg["split_regex"]:
            article_sentences = re.split(set_cfg["split_regex"], text)
        # If language is Russian, use nltk's built-in tokenizing logic for the language
        elif lang_code == "ru":
            article_sentences = sent_tokenize(text, language='russian')
        # For other languages, use nltk's default sent_tokenize logic
        else:
            article_sentences = sent_tokenize(text)

        # Check whether delineated sentences are valid
        # Initialize list that will contain all valid_sentences from dataset entry's text
        valid_sentences = []
        # Iterate through every sentence in the dataset entry
        for s in article_sentences:
            # Remove extra whitespace
            s_stripped = s.strip()

            # Skip and remove sentences with newlines because these are likely to be metadata or invalid sentences
            if not s_stripped or '\n' in s_stripped or '\r' in s_stripped:
                continue

            # Count sentence length and include only those that meet min_length
            if set_cfg["use_words"]:
                count = len([w for w in s_stripped.split() if w])
            else:
                count = len(s_stripped)
            
            if count < set_cfg["min_len"]:
                continue

            # Retain only sentences that start with capital letters
            if lang_code == "hi":
                if not re.match(r'[\u0900-\u097F]', s_stripped): continue
            elif lang_code == "ru":
                if not re.match(r'[А-ЯЁ]', s_stripped): continue
            elif lang_code == "zh":
                pass
            else:
                if not s_stripped[0].isupper(): continue

            # Retain only sentences that end wih proper punctuation
            if not s_stripped.endswith(set_cfg["punct"]):
                continue

            # Add sentences that meet above conditions to valid_sentences
            valid_sentences.append(s_stripped)

        # If there are valid sentences, randomly choose one and append to (extracted) sentences
        if valid_sentences:
            sentences.append(random.choice(valid_sentences))
            pbar.update(1) # Update bar only when a sentence is successfully added

    pbar.close()

    # Save to file
    with open(output_file, "w", encoding="utf-8-sig") as f:
        for s in sentences[:num_sentences]:
            f.write(s + "\n")

    print(f"\nSuccess! Saved to {output_file}")

if __name__ == "__main__":
    args = get_args()
    extract_sentences(args.lang, args.num, args.out)