import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu


def minibatches(inputs, targets, minibatch_size):
    """ batch generator. yields x and y batch. """
    x_batch, y_batch = [], []
    for inp, tgt in zip(inputs, targets):
        if len(x_batch) == minibatch_size and len(y_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch.append(inp)
        y_batch.append(tgt)

    if len(x_batch) != 0:
        for inp, tgt in zip(inputs, targets):
            if len(x_batch) != minibatch_size:
                x_batch.append(inp)
                y_batch.append(tgt)
            else:
                break
        yield x_batch, y_batch


def pad_sequences(sequences, pad_tok, tail=True):
    """ pads the sentences, so that all sentences in a batch have the same length. """
    max_length = max(len(x) for x in sequences)

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        if tail:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        else:
            seq_ = [pad_tok] * max(max_length - len(seq), 0) + seq[:max_length]

        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def reset_graph(seed=97):
    """ helper function to reset the default graph. this often
        comes handy when using jupyter noteboooks.
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



def sample_results(preds, en_ind2word, de_ind2word, en_word2ind, de_word2ind, de_inds, en_inds):
    beam = False

    if len(np.array(preds).shape) == 4:
        beam = True

    '''
        I dont know if this is the right way to use bleu score, but i append each calculated bleu score
        to this array and then in the end compute the mean bleu score?
        sees logical
    '''
    bleu_scores = []

    for pred, de, en, seq_length in zip(preds[0],
                                        de_inds,
                                        en_inds,
                                        [len(inds) for inds in de_inds]):
        print('\n\n\n', 100 * '-')

        if beam:
            actual_text = [en_ind2word[word] for word in reversed(en) if 
                           word != en_word2ind["<s>"] and word != en_word2ind["</s>"]]
            actual_translation = [de_ind2word[word] for word in de if
                                  word != de_word2ind["<s>"] and word != de_word2ind["</s>"]]
            created_translation = []
            for word in pred[:seq_length]:
                if word[0] != de_word2ind['</s>'] and word[0] != de_word2ind['<s>']:
                    created_translation.append(de_ind2word[word[0]])
                    continue
                else:
                    continue

            bleu_score = sentence_bleu([actual_translation], created_translation)
            bleu_scores.append(bleu_score)

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual translation:\n{}\n'.format(' '.join(actual_translation)))
            print('Created translation:\n{}\n'.format(' '.join(created_translation)))
            print('Bleu-score:', bleu_score)
            print()


        else:

            actual_text = [en_ind2word[word] for word in reversed(en) if
                           word != en_word2ind["<s>"] and word != en_word2ind["</s>"]]
            actual_translation = [de_ind2word[word] for word in de if word != de_word2ind["<s>"] and word != de_word2ind["</s>"]]
            created_translation = [de_ind2word[word] for word in pred if word != de_word2ind["<s>"] and word != de_word2ind["</s>"]][:seq_length]
            bleu_score = sentence_bleu([actual_translation], created_translation)
            bleu_scores.append(bleu_score)

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual translation:\n{}\n'.format(' '.join(actual_translation)))
            print('Created translation:\n{}\n'.format(' '.join(created_translation)))
            print('Bleu-score:', bleu_score)

    bleu_score = np.mean(bleu_scores)
    print('\n\n\nTotal Bleu Score:', bleu_score)


