import nltk
from collections import Counter

def preprocess(text, language='english', lower=True):
    """Tokenize and lower the text"""
    words = []
    tokenized_text = []

    for line in text:
        tokenized = nltk.word_tokenize(line, language=language)
        if lower:
            tokenized = [word.lower() for word in tokenized]
        tokenized_text.append(tokenized)
        for word in tokenized:
            words.append(word)

    most_common = Counter(words).most_common()

    return tokenized_text, most_common


def create_vocab(most_common_words, specials, threshold=0):
    """creates lookup dicts."""
    word2ind = {}
    ind2word = {}
    i = 0

    for sp in specials:
        word2ind[sp] = i
        ind2word[i] = sp
        i += 1

    for word, freq in most_common_words:
        if freq >= threshold:
            word2ind[word] = i
            ind2word[i] = word
            i += 1

    assert len(word2ind) == len(ind2word)

    return word2ind, ind2word, len(word2ind)


def convert_to_inds(text,
                    word2ind,
                    reverse=False,
                    eos=False,
                    sos=False):
    """ converts the given input to int values corresponding to the given word2ind
        if set to True reverse the sequence. if true append eos. if true insert sos in the beginning .
    """
    unknown_words = set()
    text_inds = []

    for sentence in text:
        sentence_inds = []
        for word in sentence:
            if word in word2ind.keys():
                sentence_inds.append(word2ind[word])
            else:
                sentence_inds.append(word2ind['<unk>'])
                unknown_words.update(word)

        if reverse:
            sentence_inds = list(reversed(sentence_inds))
        if eos:
            sentence_inds.append(word2ind['</s>'])
        if sos:
            sentence_inds.insert(0, word2ind['<s>'])
        text_inds.append(sentence_inds)

    return text_inds, unknown_words


def convert_to_words(sentence, ind2word):
    """back to words"""
    return [ind2word[word] for word in sentence]