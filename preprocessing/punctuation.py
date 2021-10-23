import string

def toclean_text(text):
    '''Remove punctuations from text'''
    clean_text = [char for char in text if char not in string.punctuation]
    clean_text = ''.join(clean_text)

    return clean_text