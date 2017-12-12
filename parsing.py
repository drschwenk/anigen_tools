import os
import spacy
import dill
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tree import ParentedTree


# nlp = spacy.load('en')
# core_nlp_base = '/Users/schwenk/wrk/animation_gan/phrase_cues/deps/stanford_core_nlp/stanford-corenlp-full-2017-06-09/'
#
# parser = StanfordParser(path_to_jar=core_nlp_base + 'stanford-corenlp-3.8.0.jar',
#                         path_to_models_jar=core_nlp_base +'stanford-corenlp-3.8.0-models.jar')

const_parse_path = '/Users/schwenk/wrk/animation_gan/build_dataset/dataset'
const_parse_dir = 'const_parses'


def const_parse(doc, parser):
    raw_sentences = sent_tokenize(doc)
    sentences = [' '.join(word_tokenize(s)) for s in raw_sentences]
    sent_parses = [list(i)[0] for i in parser.raw_parse_sents(sentences)]
    return sent_parses


def write_tree(custom_tree, vid, sent_n):
    out_file = os.path.join(const_parse_dir, '{}_sent_{}.pkl'.format(vid, sent_n))
    out_path = os.path.join(const_parse_path, out_file)
    custom_tree.pickle_self(out_path)
    return out_file


def format_parse(parsed_sentences):
    parented_trees = [ParentedTree.convert(sent) for sent in parsed_sentences]
    custom_trees = [Tree(tree) for tree in parented_trees]
    return custom_trees


def save_video_cont_parses(custom_parse_trees, vid):
    saved_paths = [write_tree(tree, vid, idx) for idx, tree in enumerate(custom_parse_trees)]
    return saved_paths


def parse_description(vid_text, g_vid, nlp, parser):
    vid_text = vid_text.replace('  ', ' ')
    doc = nlp(vid_text)
    noun_phrase_chunks = {
        'chunks': [(np.start, np.end) for np in doc.noun_chunks],
        'named_chunks': [np.text for np in doc.noun_chunks]
    }
    constituent_parse = const_parse(vid_text, parser)
    pos_tags = [sent.pos() for sent in constituent_parse]
    # parse_trees = format_parse(constituent_parse)
    # parse_file_paths = save_video_cont_parses(parse_trees, g_vid)
    parses = {
                'noun_phrase_chunks': noun_phrase_chunks,
                'pos_tags': pos_tags,
                # 'constituent_parse': parse_file_paths
             }

    return parses, constituent_parse


def parse_video(video, nlp, parser):
    vid_parse, cons_parse = parse_description(video.description(), video.gid(), nlp, parser)
    video._data['parse'] = vid_parse
    return cons_parse


class Tree(object):
    def __init__(self, nltk_parented_tree):
        self.subtrees = []
        self._node_lookup = {}
        self.leaves = []
        for node in list(nltk_parented_tree.subtrees()):
            nkey = str(node.treeposition())
            if len(node.leaves()) > 1 or len(list(node.subtrees())) > 1:
                self.subtrees.append(Node(node))
            else:
                node.set_label(' '.join([node.label(), node.leaves()[0]]))
                leaf_node = Node(node)
                self.subtrees.append(leaf_node)
                self.leaves.append(leaf_node)
            self._node_lookup[nkey] = self.subtrees[-1]

        self.root_node = [node for node in self.subtrees if node.value == 'ROOT'][0]
        self.word_pos_to_node = {idx: node for idx, node in enumerate(self.leaves)}

        for node in self.subtrees:
            node.left_sibling = self._node_lookup.get(node.left_sibling)
            node.right_sibling = self._node_lookup.get(node.right_sibling)
            node.parent = self._node_lookup.get(node.parent)
            node.children = [self._node_lookup.get(cn) for cn in node.children]

    def pickle_self(self, out_path):
        with open(out_path, 'wb') as f:
            dill.dump(self, f, byref=False)


class Node(object):
    def __init__(self, nltk_node):
        self.value = nltk_node.label()  # (tag,word/phrase)
        self.left_sibling = Node.get_tree_position(nltk_node.left_sibling())  # Instance of class Node
        self.right_sibling = Node.get_tree_position(nltk_node.right_sibling())  # Instance of class Node
        self.parent = Node.get_tree_position(nltk_node.parent())  # Instance of class Node
        child_subtrees = list(nltk_node.subtrees())
        if len(child_subtrees) > 1:
            self.children = [str(tree.treeposition()) for tree in child_subtrees if tree.parent() == child_subtrees[0]]
        else:
            self.children = []

    @classmethod
    def get_tree_position(cls, node):
        if node:
            return str(node.treeposition())
        else:
            return

