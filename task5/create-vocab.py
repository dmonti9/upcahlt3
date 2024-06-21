import os
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


def extract_text(xml_path):
    tree = parse(xml_path)
    sentences = tree.getElementsByTagName("sentence")
    texts = [s.getAttribute("text") for s in sentences]
    return texts


def build_vocabulary(data_dirs, output_file):
    words = set()
    for dir_path in data_dirs:
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            if os.path.isfile(full_path):
                texts = extract_text(full_path)
                for text in texts:
                    tokens = word_tokenize(text)
                    words.update(tokens)
    # Save the vocabulary to a file
    with open(output_file, "w", encoding="utf-8") as f:
        for word in sorted(words):  # Sorting is optional but helps in readability
            f.write(word + "\n")
    return words


# Specify your data directories and output file here
data_dirs = ["../data/devel", "../data/test", "../data/train"]
output_file = "vocabulary.txt"
vocabulary = build_vocabulary(data_dirs, output_file)

print("Vocabulary Size:", len(vocabulary))
print("Vocabulary saved to:", output_file)
