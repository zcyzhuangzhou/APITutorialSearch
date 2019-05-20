import pickle
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from sekg.util.url_util import URLUtil

# re connect
from sekg.wiki.WikiDataItem import WikiDataItem
from sekg.wiki.search_domain_wiki.wiki_info_model import WikiInfo


def conn_try_again(function):
    retries = 5
    # retry time
    count = {"num": retries}

    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as err:
            if count['num'] < 2:
                count['num'] += 1
                return wrapped(*args, **kwargs)
            else:
                raise Exception(err)

    return wrapped


class WikiTool:
    def __init__(self, domain_name_set_to_search, use_proxies=False, max_candidate_num=3):
        self.title_2_wiki = dict()
        self.wiki_data_id_2_wikidata = dict()
        self.domain_name_set_to_search = domain_name_set_to_search
        self.use_proxies = use_proxies
        self.max_candidate_num = max_candidate_num

    def save(self, save_path):

        with open(str(Path(save_path) / "title_2_wiki_dict"), 'wb') as f:
            pickle.dump(self.title_2_wiki, f, pickle.HIGHEST_PROTOCOL)

        with open(str(Path(save_path) / "wiki_data_id_2_wikidata_dict"), 'wb') as f:
            pickle.dump(self.wiki_data_id_2_wikidata, f, pickle.HIGHEST_PROTOCOL)

    def load(self, save_path):
        with open(str(Path(save_path) / "title_2_wiki_dict"), 'rb') as f:
            self.title_2_wiki = pickle.load(f)
        with open(str(Path(save_path) / "wiki_data_id_2_wikidata_dict"), 'rb') as f:
            self.wiki_data_id_2_wikidata = pickle.load(f)

    def get_wiki_id(self, wd_url):
        return (str(wd_url).split('/'))[-1]

    def find(self, title):
        if title in self.title_2_wiki:
            return self.title_2_wiki[title]
        else:
            return self.search_title(title)

    def search_title(self, title):
        wiki_info_list = []
        wiki_id_list, wikipedia_url_list, wikipedia_title_list, wd_item_list = (self.get_wd_by_title(title))

        for wiki_id, wikipedia_url, wikipedia_title, wd_item in zip(wiki_id_list, wikipedia_url_list,
                                                                    wikipedia_title_list, wd_item_list):
            wiki_info = WikiInfo(wiki_data_id=wiki_id, wikipedia_url=wikipedia_url, wikipedia_title=wikipedia_title,
                                 wiki_data_item=wd_item)
            wiki_info_list.append(wiki_info)

        self.title_2_wiki[title] = wiki_info_list
        return wiki_info_list

    def start_search(self):
        for title in self.domain_name_set_to_search:
            self.search_title(title)

    @conn_try_again
    def get_wd_by_title(self, title):
        wiki_data_id_list = []
        wikipedia_url_list = []
        wikipedia_title_list = []
        if self.use_proxies:
            proxies = {"http": "127.0.0.1:1080", "https": "127.0.0.1:1080"}
        else:
            proxies = None
        try:
            base_url = "https://en.wikipedia.org"
            url = base_url + "/wiki/" + title
            m = requests.get(url, proxies=proxies)
            if m.status_code == 200:
                soup = BeautifulSoup(m.content, "lxml")
                wikidata_node_list = soup.select('#t-wikibase')
                if wikidata_node_list:
                    wikidata_node = wikidata_node_list[0]
                    for link in wikidata_node.findAll('a', attrs={'href': re.compile("^https://")}):
                        wd_href = link.get('href')
                        wd_id = self.get_wiki_id(wd_href)
                        wikipedia_title = URLUtil.parse_url_to_title(m.url)

                        wiki_data_id_list.append(wd_id)
                        wikipedia_url_list.append(m.url)
                        wikipedia_title_list.append(wikipedia_title)
                else:
                    print("wikidata_node_list is empty")
            else:
                new_title = title.replace(" ", "+")
                url = "https://en.wikipedia.org/w/index.php?search=" + new_title + "&title=Special%3ASearch&go=Go"
                m = requests.get(url, proxies=proxies)
                if m.status_code == 200:
                    soup = BeautifulSoup(m.content, "lxml")
                    candidate_list = soup.select('.mw-search-result-heading')
                    for i, candidate in enumerate(candidate_list):
                        if i == self.max_candidate_num - 1:
                            break
                        for link in candidate.findAll('a', attrs={'href': re.compile("^/wiki/")}):
                            wd_href = link.get('href')
                            redirect_url = base_url + wd_href
                            redirect_m = requests.get(redirect_url, proxies=proxies)
                            if redirect_m.status_code == 200:
                                soup = BeautifulSoup(redirect_m.content, "lxml")
                                wikidata_node_list = soup.select('#t-wikibase')
                                if wikidata_node_list:
                                    wikidata_node = wikidata_node_list[0]
                                    for link in wikidata_node.findAll('a', attrs={'href': re.compile("^https://")}):
                                        wd_href = link.get('href')
                                        wd_id = self.get_wiki_id(wd_href)
                                        if len(wd_id) > 1 and wd_id[0] == "Q":
                                            wiki_data_id_list.append(wd_id)
                                            wikipedia_url_list.append(redirect_m.url)
                                            wikipedia_title_list.append(URLUtil.parse_url_to_title(redirect_m.url))
                                            break
                else:
                    print("fail to find")
            wd_item_list = []
            for wiki_data_id in wiki_data_id_list:
                if wiki_data_id not in self.wiki_data_id_2_wikidata:
                    wd_item = WikiDataItem(wiki_data_id)
                    if wd_item:
                        self.wiki_data_id_2_wikidata[wiki_data_id] = wd_item
                else:
                    wd_item = self.wiki_data_id_2_wikidata[wiki_data_id]
                wd_item_list.append(wd_item)

            return wiki_data_id_list, wikipedia_url_list, wikipedia_title_list, wd_item_list
        except Exception as e:
            print(e)


if __name__ == '__main__':
    test_set = set()
    test_set.add("Peer alarm")
    wiki_tool = WikiTool(domain_name_set_to_search=test_set, use_proxies=False, max_candidate_num=5)
    wiki_tool.start_search()
    p = "D:\\实验室\\src\\save"
    wiki_tool.save(p)
    w = WikiTool(domain_name_set_to_search=set(), use_proxies=False, max_candidate_num=5)
    w.load(p)
    print(w.title_2_wiki)
