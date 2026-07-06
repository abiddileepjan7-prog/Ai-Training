import string


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove leading and trailing spaces
    text = text.strip()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = " ".join(text.split())

    return text