# -*- coding: utf-8 -*-
import json
import traceback

import requests


class WikiDataItem:
    def __init__(self, wd_item_id, init_at_once=True):
        self.wd_item_id = wd_item_id
        self.property_name_list = None
        self.data_dict = None
        self.relation_property_name_list = None
        self.source_wd_dict_json = None
        self.is_init = False
        if init_at_once:
            self.init_wikidata_item(self.wd_item_id)

    def get_en_name(self):
        if self.source_wd_dict_json:
            if 'labels' in self.source_wd_dict_json.keys() and "en" in self.source_wd_dict_json['labels'].keys():
                return self.source_wd_dict_json['labels']["en"]["value"]
        return None

    @staticmethod
    def __merge_value_from_dict_list(dict_list, value_key='value'):
        values = []
        for d in dict_list:
            if value_key in d:
                values.append(d[value_key])
        # return ',,'.join(values)
        return values

    def __parse_wd_json_to_dict(self, wd_item_json):
        wd_item_property_dict = {}
        self.__extract_muilti_language_property(wd_item_json, 'aliases', wd_item_property_dict)
        self.extract_muilti_language_property_from_dict(wd_item_json, 'labels', wd_item_property_dict)
        self.extract_muilti_language_property_from_dict(wd_item_json, 'descriptions', wd_item_property_dict)

        self.__extract_sitelinks(wd_item_json, wd_item_property_dict)
        property_name_list, relation_property_name_list = WikiDataItem.__extract_claims(wd_item_json=wd_item_json,
                                                                                        wd_item_property_dict=wd_item_property_dict)
        if 'labels_en' in wd_item_property_dict:
            wd_item_property_dict['name'] = wd_item_property_dict['labels_en']
        return wd_item_property_dict, property_name_list, relation_property_name_list

    @staticmethod
    def __extract_metadata(entity_metadata):
        metadata_dict = {}
        metadata_dict['wd_item_id'] = entity_metadata['id']
        metadata_dict['lastrevid'] = entity_metadata['lastrevid']
        metadata_dict['modified'] = entity_metadata['modified']
        return metadata_dict

    @staticmethod
    def __extract_sitelinks(wd_item_json, wd_item_property_dict):
        if 'sitelinks' in wd_item_json.keys():
            sitelinks = wd_item_json['sitelinks']
            if not sitelinks:
                return
            for k, v in sitelinks.items():
                k_title = "site:" + k
                wd_item_property_dict[k_title] = v['url']

    @staticmethod
    def __extract_claims(wd_item_json, wd_item_property_dict):
        claims_dict = wd_item_json['claims']
        if not claims_dict:
            return [], []
        property_name_list = []
        relation_property_name_list = []
        for k, v in claims_dict.items():
            property_name_list.append(k)
            wd_item_property_dict[k] = WikiDataItem.__extract_claims_property_from_value_list(v)
            if v and WikiDataItem.is_property_relation(v[0]):
                relation_property_name_list.append(k)
        return property_name_list, relation_property_name_list

    @staticmethod
    def __extract_claims_property_from_value_list(value_list):
        extract_values = []
        for value in value_list:
            t = WikiDataItem.__extract_claims_property_item_from_value_list(value)
            if t:
                extract_values.append(t)
        # return ',,'.join(extract_values)
        return extract_values

    @staticmethod
    def is_property_relation(value_dict):
        try:
            mainsnak = value_dict['mainsnak']
            if mainsnak['snaktype'] == 'novalue' or mainsnak['snaktype'] == 'somevalue':
                return False
            data_type = mainsnak['datatype']
            if data_type == 'wikibase-item':
                return True
            else:
                return False
        except Exception as e:
            traceback.print_exc()
            return False

    @staticmethod
    def __extract_claims_property_item_from_value_list(value_dict):
        mainsnak = value_dict['mainsnak']
        if mainsnak['snaktype'] == 'novalue' or mainsnak['snaktype'] == 'somevalue':
            return None
        data_type = mainsnak['datatype']
        value = ''

        string_value_type = ['string', 'external-id', 'math', 'url', 'commonsMedia', 'geo-shape']

        if data_type in string_value_type:
            value = mainsnak['datavalue']['value']
        if data_type == 'wikibase-item':
            value = mainsnak['datavalue']['value']['id']
        if data_type == 'wikibase-property':
            value = 'P' + str(mainsnak['datavalue']['value']['numeric-id'])
        if data_type == 'time':
            value = mainsnak['datavalue']['value']['time']
        if data_type == 'monolingualtext':
            value = mainsnak['datavalue']['value']['text']
        if data_type == 'quantity':
            value = mainsnak['datavalue']['value']['amount']
        if data_type == 'globe-coordinate':
            value = str(mainsnak['datavalue']['value']['latitude']) + ',' + str(
                mainsnak['datavalue']['value']['longitude'])

        return value

    def __extract_muilti_language_property(self, wd_item_json, key, wd_item_property_dict):
        language_dict = wd_item_json[key]
        if not language_dict:
            return
        for k, v in language_dict.items():
            try:
                full_property_name = key + '_' + k
                wd_item_property_dict[full_property_name] = self.__merge_value_from_dict_list(v)
            except Exception as error:
                print(error)

    @staticmethod
    def extract_muilti_language_property_from_dict(wd_item_json, key, wd_item_property_dict):
        language_dict = wd_item_json[key]
        if not language_dict:
            return
        for k, v in language_dict.items():
            try:
                full_property_name = key + '_' + k
                wd_item_property_dict[full_property_name] = v['value']
            except Exception as error:
                print(error)

    def init_wikidata_item_from_json_string(self, data_json_string):
        try:

            dict_json = self.parse_illegal_json_string(data_json_string)

            self.source_wd_dict_json = dict_json
            self.wd_item_id = dict_json["id"]
            self.__parse()
            self.is_init = True
            return self
        except Exception as error:
            traceback.print_exc()
            self.source_wd_dict_json = None
            self.is_init = False
            return self

    def init_wikidata_item(self, wd_item_id):
        try:
            r = requests.get(
                "https://www.wikidata.org/wiki/Special:EntityData/{wd_item_id}.json".format(wd_item_id=wd_item_id))
            json_response = r.content.decode(encoding='utf-8')
            dict_json = json.loads(json_response, encoding='utf-8')
            self.source_wd_dict_json = dict_json["entities"][wd_item_id]
            self.wd_item_id = wd_item_id
            self.__parse()
            self.is_init = True
            return self
        except Exception as error:
            traceback.print_exc()
            self.source_wd_dict_json = None
            self.is_init = False
            return self

    def init_wikidata_item_from_wikipedia_title(self, wikipedia_title):
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&format=json&titles=" + wikipedia_title)

            json_response = r.content.decode(encoding='utf-8')
            query_json = json.loads(json_response, encoding='utf-8')
            wd_item_id = None
            pages = query_json["query"]["pages"]
            for page_index, page in pages.items():
                if page_index == "-1":
                    break
                wd_item_id = page["pageprops"]["wikibase_item"]
            self.wd_item_id = wd_item_id
            if wd_item_id is None:
                self.is_init = False
                return self
            return self.init_wikidata_item(wd_item_id=wd_item_id)

        except Exception as error:
            traceback.print_exc()
            self.source_wd_dict_json = None
            self.is_init = False
            return self

    def init_wikidata_item_from_wikipedia_url(self, wikipedia_url):
        wikipedia_title = wikipedia_url.replace("https://en.wikipedia.org/wiki/", "")
        return self.init_wikidata_item_from_wikipedia_title(wikipedia_title)

    def __parse_wikidata_item_to_dict(self):
        property_dict, property_name_list, relation_property_name_list = self.__parse_wd_json_to_dict(
            self.source_wd_dict_json)
        metadata_dict = self.__extract_metadata(self.source_wd_dict_json)

        return dict(property_dict, **metadata_dict), property_name_list, relation_property_name_list

    def __parse(self):
        data_dict, property_name_list, relation_property_name_list = self.__parse_wikidata_item_to_dict()
        self.data_dict = data_dict
        self.property_name_list = property_name_list
        self.relation_property_name_list = relation_property_name_list

    def get_wikidata_item_property_dict(self):
        return self.data_dict

    def get_relation_property_name_list(self):
        return self.relation_property_name_list

    def get_wikidata_item_property_name_list(self):
        return self.property_name_list

    def exist(self):
        return self.is_init

    def get_en_wiki_url(self):
        pass

    def get_en_wiki_title(self):
        try:
            if self.source_wd_dict_json:
                if "sitelinks" in self.source_wd_dict_json.keys() and "enwiki" in self.source_wd_dict_json[
                    'sitelinks'].keys():
                    return self.source_wd_dict_json['sitelinks']["enwiki"]["title"]
        except Exception:
            traceback.print_exc()
        return None

    @staticmethod
    def parse_illegal_json_string(data_json_string):

        if data_json_string == None or data_json_string == "":
            return None

        try:
            try:
                json_instance = json.loads(data_json_string, encoding='utf-8')
                return json_instance
            except Exception:
                data_json_string = json.dumps(eval(data_json_string))

                json_instance = json.loads(data_json_string, encoding='utf-8')
                return json_instance
        except Exception:
            traceback.print_exc()
            return None

    @staticmethod
    def is_valid_json_string(data_json_string):
        try:
            json_instance = WikiDataItem.parse_illegal_json_string(data_json_string)
            if json_instance == None:
                return False
            if json_instance["claims"] == {}:
                return False
            return True
        except Exception:
            traceback.print_exc()
            return False
