import spacy

from .base import Preprocessor

nlp = spacy.load('en_core_web_sm')


class SpacyTextPreprocessor(Preprocessor):
    def extract_words_for_query(self, query):
        np_v_list = []

        doc = nlp(query)
        for np in doc.noun_chunks:
            np_words = []
            for np_token in np:
                lemma_token = np_token.lemma_
                if np_token.is_stop:
                    continue
                np_words.append(lemma_token)

            clean_np_str = " ".join(np_words)
            if clean_np_str:
                np_v_list.append(clean_np_str)

        for token in doc:
            if token.pos_ == "VERB" and token.is_stop == False:
                verb_lemma = token.lemma_
                np_v_list.append(verb_lemma)

        # todo: extract CamelName split as keyword from this

        return set(" ".join(np_v_list).split())

    def clean(self, text):
        """
        return a list of token from the text, only remove stopword and lemma the word
        :param text: the text need to preprocess
        :return: list of str
        """

        result = []
        doc = nlp(text)

        for token in doc:
            if token.pos_ == "PUNCT":
                continue
            if token.is_stop:
                continue

            result.append(token.lemma_.lower())

        return result
