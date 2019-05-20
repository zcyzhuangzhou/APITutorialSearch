#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re

import gensim
from bs4 import BeautifulSoup


class CodeTextPreprocessor:
    pattern = re.compile(r'\s+')
    CODE_FRAGMENT_MARK = "__CODE__"

    def __clean_format(self, text):
        """
        clean text format for text extract from html
        :param text:
        :return:
        """
        return re.sub(self.pattern, " ", text.replace('\n', ' ').replace(u'\u00a0', " "))

    def clean_html_text(self, html_text):
        if html_text is None or len(html_text) == 0:
            return ""

        soup = BeautifulSoup(html_text, "lxml")
        codeTags = soup.find_all(name=["pre", 'blockquote'])

        for tag in codeTags:
            tag.string = " " + self.CODE_FRAGMENT_MARK + " . \n"

        ## todo: this is for <li> <ul>,
        # list_groups = soup.find_all(name=["ol", "ul"])
        # for list_group in list_groups:
        #     list_items = list_group.find_all("li")
        #     num = 1
        #     for item in list_items:
        #         item.string = "{0}.{1}\n.".format(str(num), item.string)
        #         num = num + 1
        #
        # # todo: the sentence may lack of Punctuation mark in every <p> tag end. it will be

        cleanText = soup.get_text()
        decode_clean_text = gensim.utils.decode_htmlentities(cleanText)
        ## todo: test the code
        return self.__clean_format(decode_clean_text)
