from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
import string
import re

punct_set = set(string.punctuation)
punct_set.remove('.')
punkt_param = PunktParameters()
punkt_param.abbrev_types = {'dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'ms'}
sentence_splitter = PunktSentenceTokenizer(punkt_param)

main_characters_lower = {
    "fred": 'Fred',
    "wilma": 'Wilma',
    "mr slate": "Mr. Slate",
    "barney": "Barney",
    "betty": "Betty",
    "pebbles": "Pebbles",
    "dino": "Dino",
    "baby puss": "Baby Puss",
    "hoppy": "Hoppy",
    "bamm bamm": "Bamm Bamm"
}

# def const_parse(doc, parser):
#     raw_sentences = sentence_splitter.tokenize(doc)
#     sentences = [' '.join([w for w in wordpunct_tokenize(s) if set(w) - punct_set]).replace(' .',  '.') for s in raw_sentences]
#     sent_parses = [list(i)[0] for i in parser.raw_parse_sents(sentences)]
#     return sent_parses


def check_sub_subtrees(subtree):
    for tree in list(subtree.subtrees())[1:]:
        if tree.label() in ['NP']:
            return False
    return True


def apply_fixes(raw_str):
    raw_str = raw_str.replace(' \'', '\'')
    return raw_str


def extract_np(psent):
    for subtree in psent.subtrees():
        if subtree.label() == 'NP' and check_sub_subtrees(subtree):
            subprod = subtree.productions()[0].unicode_repr()
            if 'NN' in subprod or 'NNP' in subprod:
                if 'CC' not in subprod:
                    yield ' '.join(word for word in subtree.leaves()).replace(' \'', '\'')
                else:
                    for st in subtree.subtrees():
                        if st.label() in ['NNP', 'NN']:
                            yield st.leaves()[0]


def compute_token_spans(const_parse_sents, txt):
    offset = 0
    for const_parse_sent in const_parse_sents:
        tokens = const_parse_sent.leaves()
        for token in tokens:
            offset = txt.find(token, offset)
            yield token, offset, offset + len(token)
            offset += len(token)


def assign_word_spans(noun_phrases_w_spans, doc, token_spans):
    chunk_spans = []
    seen_chunks = []
    for np in noun_phrases_w_spans:
        # print(np)
        char_spans = [(m.start(), m.end() - 1) for m in re.finditer(np + '\s|' + np + '\.', doc)]
        # print(seen_chunks)
        occ_n = seen_chunks.count(np)
        # print(occ_n)
        # print(char_spans)
        start, end = char_spans[occ_n]
        start_w, end_w = None, None
        for w_idx, token_span in enumerate(token_spans):
            token, ts, te = token_span
            if ts == start:
                start_w = w_idx
            if te == end:
                end_w = w_idx + 1
        if type(start_w) == int and type(end_w) == int:
            chunk_spans.append([start_w, end_w])
        else:
            print(np)
            print('failed')
            raise IndexError
        np_pieces = np.split()
        seen_chunks += list(set(np_pieces).union(set([np])))
    return chunk_spans


def np_chunker(doc, parsed_sents):
    recovered_tokens = ' '.join([item for sublist in parsed_sents for item in sublist.leaves()]).replace(' .',  '.')
    noun_phrases = [list(extract_np(sent)) for sent in parsed_sents]
    # print(noun_phrases)
    noun_phrases = [item for sublist in noun_phrases for item in sublist]
    #     noun_phrase_spans = [list(extract_np_spans(doc, sent)) for sent in noun_phrases]
    token_spans = list(compute_token_spans(parsed_sents, recovered_tokens))
    # print(list(token_spans))
    noun_phrase_spans = assign_word_spans(noun_phrases, recovered_tokens, token_spans)
    return {'chunks': noun_phrase_spans, 'named_chunks': noun_phrases, 'token_spans': token_spans,
            'aligned_description': recovered_tokens}


def sanitize_text(d_text):
    d_text = ' '.join(d_text.split())
    d_text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', d_text)
    if d_text[-1] != '.':
        d_text += '.'
    for lc, uc in main_characters_lower.items():
        d_text = d_text.replace(lc, uc)
    return d_text


def parse_description(vid_text, nlp, parser):
    vid_text = sanitize_text(vid_text)
    raw_sentences = sentence_splitter.tokenize(vid_text)
    try:
        sentences = [' '.join([w for w in wordpunct_tokenize(s) if set(w) - punct_set]).replace(' .',  '.') for s in raw_sentences]
        # sentences = raw_sentences
        # print('here', sentences)
        # docs = [nlp(sent) for sent in sentences]

        # noun_phrase_chunks = {
        #     'chunks': [[(np.start, np.end) for np in doc.noun_chunks] for doc in docs],
        #     'named_chunks': [[np.text for np in doc.noun_chunks] for doc in docs]
        # }
        # constituent_parse = const_parse(vid_text, parser)
        constituent_parse = [list(i)[0] for i in parser.raw_parse_sents(sentences)]
        # return constituent_parse
        # print([s.leaves() for s in constituent_parse])
        noun_phrase_chunks = np_chunker(vid_text, constituent_parse)
    except IndexError:
        # sentences = [' '.join([w for w in word_tokenize(s) if set(w) - punct_set]).replace(' .',  '.') for s in raw_sentences]
        constituent_parse = [list(i)[0] for i in parser.raw_parse_sents(raw_sentences)]
        noun_phrase_chunks = np_chunker(vid_text, constituent_parse)
    pos_tags = [sent.pos() for sent in constituent_parse]
    # pos_tags = [(token.text, token.pos_, token.string) for token in doc]
    pos_tags = [item for sublist in pos_tags for item in sublist]
    parses = {
        'noun_phrase_chunks': noun_phrase_chunks,
        'pos_tags': pos_tags,
    }
    return parses


def parse_video(video, nlp, parser):
    # print(video.gid())
    vid_parse = parse_description(video.description(), nlp, parser)
    video._data['parse'] = vid_parse

