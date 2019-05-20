class Preprocessor:
    """
    a basic text preprocessor
    """

    def extract_words_for_query(self, text):
        """
        extract the keyword set for a query text
        :param text:
        :return:
        """
        return set(self.clean(text))

    def clean(self, text):
        """
        return a list of token from the text
        :param text: the text need to preprocess
        :return: list of str
        """
        return text.split()


class SimplePreprocessor(Preprocessor):
    remove_list = {"\n", "\t", "\r", "/", "*", ".", ";", "@", "{", "}", "<p>", "(", ")", "#", "=", ":", "+", "-",
                   "!",
                   "[", "]", ",", ":", "<", ">", "|", "\\", "&", "'", "?", "<", ">"}

    def remove_special_char(self, str):
        """
        input;str
        :return: str
        """
        new_str = str
        for item in self.remove_list:
            new_str = new_str.replace(item, " ").replace("  ", " ")
        return new_str.strip()

    def clean(self, text):
        """
        return a list of token from the text, only remove special char and lower the case
        :param text: the text need to preprocess
        :return: list of str
        """
        text = self.remove_special_char(text)
        return text.lower().split()
