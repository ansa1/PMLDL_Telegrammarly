from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
import src.bot_states as bot_states
import src.bot_messages as bot_messages
import telegram
import logging

import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertForSequenceClassification
from hunspell import Hunspell
from difflib import SequenceMatcher
import en_core_web_sm

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


def check_GE(sents):
    """Check of the input sentences have grammatical errors

    :param list: list of sentences
    :return: error, probabilities
    :rtype: (boolean, (float, float))
    """

    # Create sentence and label lists
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sents]

    tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]

    # Convert tokens to ids
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Set the maximum sequence length
    MAX_LEN = 128

    # Pad sentences to MAX_LEN
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    with torch.no_grad():
        logits = modelGED(prediction_inputs, token_type_ids=None, attention_mask=prediction_masks)

    # Move logits to CPU
    logits = logits.detach().cpu().numpy()

    # Store predictions
    predictions = []
    predictions.append(logits)

    flat_predictions = [item for sublist in predictions for item in sublist]
    prob_vals = flat_predictions
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    return flat_predictions, prob_vals


def progress_bar(some_iter):
    try:
        from tqdm import tqdm
        return tqdm(some_iter)
    except ModuleNotFoundError:
        return some_iter


modelGED = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# restore model
# TODO remove map_location param
modelGED.load_state_dict(torch.load('bert-based-uncased-GED.pth', map_location=torch.device('cpu')))

model = BertForMaskedLM.from_pretrained('bert-large-uncased')

tokenizerLarge = BertTokenizer.from_pretrained('bert-large-uncased')

gb = Hunspell("en_GB-large", hunspell_data_dir=".")

# List of common determiners
# det = ["", "the", "a", "an"]
det = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
       'her', 'its', 'our', 'their', 'all', 'both', 'half', 'either', 'neither',
       'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite']

# List of common prepositions
prep = ["about", "at", "by", "for", "from", "in", "of", "on", "to", "with",
        "into", "during", "including", "until", "against", "among",
        "throughout", "despite", "towards", "upon", "concerning"]

# List of helping verbs
helping_verbs = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'shall', 'should', 'may', 'might', 'must', 'can', 'could']


# Create a set of sentences which have possible corrected spellings
def create_spelling_set(org_text):
    sent = org_text.lower().strip().split()

    nlp = en_core_web_sm.load()
    proc_sent = nlp.tokenizer.tokens_from_list(sent)
    nlp.tagger(proc_sent)

    sentences = []

    for token in proc_sent:
        # check for spelling for alphanumeric
        if token.text.isalpha() and not gb.spell(token.text):
            new_sent = sent[:]
            # append new sentences with possible corrections
            for sugg in gb.suggest(token.text):
                new_sent[token.i] = sugg
                sentences.append(" ".join(new_sent))

    spelling_sentences = sentences

    # retain new sentences which have a minimum chance of correctness using BERT GED
    new_sentences = []

    for sent in spelling_sentences:
        _, prob_val = check_GE([sent])
        exps = [np.exp(i) for i in prob_val[0]]
        sum_of_exps = sum(exps)
        softmax = [j / sum_of_exps for j in exps]
        if softmax[1] > 0.6:
            new_sentences.append(sent)

    # if no corrections, append the original sentence
    if len(spelling_sentences) == 0:
        spelling_sentences.append(" ".join(sent))

    # eliminate duplicates
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))

    return spelling_sentences


# Create a new set of sentences with deleted determiners, prepositions & helping verbs
def create_grammar_set(spelling_sentences):
    new_sentences = []

    for text in spelling_sentences:
        sent = text.strip().split()
        for i in range(len(sent)):
            new_sent = sent[:]

            if new_sent[i] not in list(set(det + prep + helping_verbs)):
                continue

            del new_sent[i]
            text = " ".join(new_sent)

            _, prob_val = check_GE([text])
            exps = [np.exp(i) for i in prob_val[0]]
            print(prob_val)
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            if softmax[1] > 0.6:
                new_sentences.append(text)

    # eliminate duplicates
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))
    return spelling_sentences


# For each input sentence create 2 sentences:
# (1) [MASK] each word
# (2) [MASK] for each space between words
def create_mask_set(spelling_sentences):
    sentences = []

    for sent in spelling_sentences:
        sent = sent.strip().split()
        for i in range(len(sent)):
            # (1) [MASK] each word
            new_sent = sent[:]
            new_sent[i] = '[MASK]'
            text = " ".join(new_sent)
            new_sent = '[CLS] ' + text + ' [SEP]'
            sentences.append(new_sent)

            # (2) [MASK] for each space between words
            new_sent = sent[:]
            new_sent.insert(i, '[MASK]')
            text = " ".join(new_sent)
            new_sent = '[CLS] ' + text + ' [SEP]'
            sentences.append(new_sent)

    return sentences


# Check grammar
def check_grammar(org_sent, sentences, spelling_sentences):
    # what is the tokenized value of [MASK]. Usually 103
    text = '[MASK]'
    tokenized_text = tokenizerLarge.tokenize(text)
    mask_token = tokenizerLarge.convert_tokens_to_ids(tokenized_text)[0]

    new_sentences = []
    i = 0  # current sentence number
    l = len(org_sent.strip().split()) * 2  # l is no of sentencees
    mask = False  # flag indicating if we are processing space MASK

    for sent in sentences:
        i += 1

        print(".", end="")
        if i % 50 == 0:
            print("")

        # tokenize the text
        tokenized_text = tokenizerLarge.tokenize(sent)
        indexed_tokens = tokenizerLarge.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        # index of the masked token
        mask_index = (tokens_tensor == mask_token).nonzero()[0][1].item()
        # predicted token
        predicted_index = torch.argmax(predictions[0, mask_index]).item()
        predicted_token = tokenizerLarge.convert_ids_to_tokens([predicted_index])[0]

        text = sent.strip().split()
        mask_index = text.index('[MASK]')

        if not mask:
            # case of MASKed words
            mask = True
            text[mask_index] = predicted_token
            try:
                # retrieve original word
                org_word = spelling_sentences[i // l].strip().split()[mask_index - 1]
            except:
                print("!", end="")
                continue

            # use SequenceMatcher to see if predicted word is similar to original word
            if SequenceMatcher(None, org_word, predicted_token).ratio() < 0.6:
                if org_word not in list(set(det + prep + helping_verbs)) \
                        or predicted_token not in list(set(det + prep + helping_verbs)):
                    continue

            if org_word == predicted_token:
                continue
        else:
            # case for MASKed spaces
            mask = False

            # only allow determiners / prepositions  / helping verbs in spaces
            if predicted_token in list(set(det + prep + helping_verbs)):
                text[mask_index] = predicted_token
            else:
                continue

        text.remove('[SEP]')
        text.remove('[CLS]')
        new_sent = " ".join(text)

        # retain new sentences which have a minimum chance of correctness using BERT GED
        no_error, prob_val = check_GE([new_sent])
        exps = [np.exp(i) for i in prob_val[0]]
        sum_of_exps = sum(exps)
        softmax = [j / sum_of_exps for j in exps]
        if no_error and softmax[1] > 0.996:
            print("*", end="")
            new_sentences.append(new_sent)

    print("")

    # remove duplicate suggestions
    spelling_sentences = []
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))

    return spelling_sentences


print('ML INITIALIZED')
### ML ENDING

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

custom_keyboard = [['/check']]
admin_keyboard = [['/check'], ['/admin_panel']]

client_keyboard = [['/check'], ['/makeorder'], ['/back']]

standart_markup = telegram.ReplyKeyboardMarkup(custom_keyboard, resize_keyboard=True)
admin_markup = telegram.ReplyKeyboardMarkup(admin_keyboard, resize_keyboard=True)
client_markup = telegram.ReplyKeyboardMarkup(client_keyboard, resize_keyboard=True)


def first_letter_upper(s):
    return s[0].upper() + s[1:]


def start(update, context):
    update.message.reply_text(bot_messages.start_command_response, reply_markup=standart_markup)


def check(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text=bot_messages.client_response)

    return bot_states.CHECK


def ml_part(update, context):
    text = update.message.text
    print('INPUT: ' + text)

    input_sentences = []
    current_sentence = ''
    delimiters = []

    for c in text:
        if c in ['.', '!', '?']:
            input_sentences.append(current_sentence)
            delimiters.append(c)
            current_sentence = ''
        else:
            current_sentence += c

    if len(current_sentence) > 0:
        input_sentences.append(current_sentence)
        delimiters.append('.')

    corrected_text = ''

    for i in range(len(input_sentences)):
        sentence_to_correct = input_sentences[i]

        print('SENTENCE TO CORRECT: ' + sentence_to_correct)
        sentences = create_spelling_set(sentence_to_correct)
        spelling_sentences = create_grammar_set(sentences)
        sentences = create_mask_set(spelling_sentences)

        print("processing {0} possibilities".format(len(sentences)))

        sentences = check_grammar(sentence_to_correct, sentences, spelling_sentences)

        print("Suggestions & Probabilities")

        possible_corrections = "POSSIBLE CORRECTIONS:"

        if len(sentences) == 0:
            words = sentence_to_correct.split()
            words[0] = first_letter_upper(words[0])
            sentence_to_correct = ' '.join(words)
            if len(corrected_text) > 0:
                corrected_text += ' '
            corrected_text += sentence_to_correct + delimiters[i]
            continue

        _, prob_val = check_GE(sentences)

        best_match = None
        best_match_prob = 0.0

        for j in range(len(prob_val)):
            exps = [np.exp(k) for k in prob_val[j]]
            sum_of_exps = sum(exps)
            softmax = [k / sum_of_exps for k in exps]
            possible_corrections += "\n\n[{0:0.4f}] {1}".format(softmax[1] * 100, sentences[j])
            probability = softmax[1]
            if probability > best_match_prob:
                best_match = sentences[j]
                best_match_prob = probability
            print("[{0:0.4f}] {1}".format(softmax[1] * 100, sentences[j]))

        if len(corrected_text) > 0:
            corrected_text += ' '

        best_match = first_letter_upper(best_match)
        corrected_text += best_match + delimiters[i]

    print('INPUT MESSAGE: {}\nCORRECTED_MESSAGE: {}\n'.format(text, corrected_text))
    context.bot.send_message(chat_id=update.message.chat_id, text=corrected_text)

    return ConversationHandler.END


# FOR DEBUGING
def ping(update, context):
    print("context")
    print(type(context))
    print("update")
    print(update)


token = '1121924754:AAHw1itwq4GWWVSsq_mPMTnTqxQ-8ZRnwlE'
updater = Updater(token, use_context=True)
dp = updater.dispatcher


def main():
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('ping', ping))  # TODO
    check_handler = ConversationHandler(
        entry_points=[CommandHandler('check', check)],

        states={
            bot_states.CHECK: [MessageHandler(Filters.text, ml_part)]
        },

        fallbacks=[CommandHandler('start', start)]
    )
    dp.add_handler(check_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
