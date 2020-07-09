#! /usr/bin/env python

try:
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen, HTTPError

import json
import xml.etree.ElementTree as ET


class PubChemREST:
    # use this url to save cid list in eutils server
    NAMES_LISTKEY_API = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        "/name/cids/JSON?list_return=listkey"
    )

    def __init__(self, name):
        self._xml_result = None
        self.name = name

    @property
    def xml(self):
        if self._xml_result is None:
            self._xml_result = self.get_xml(self.name)
        return self._xml_result

    def get_xml(self, name, namespace="name"):

        post_body = urlencode([(namespace, name)]).encode("utf8")
        esummary = None
        try:
            response = urlopen(self.NAMES_LISTKEY_API, post_body)
            # print("successfully get list key result")
            # Construct esummary retrieve url
            lsit_key_result = json.loads(response.read())
            esummary = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                f"esummary.fcgi?db=pccompound&WebEnv="
                f"{lsit_key_result['IdentifierList']['EntrezWebEnv']}&"
                f"query_key={str(lsit_key_result['IdentifierList']['EntrezQueryKey'])}"
            )
            summary_response = urlopen(esummary)
            # print("successfully get summary result")
            # Parsing the downloaded esummary xml string
            return summary_response.read().decode("utf-8")

        except HTTPError as e:
            print(
                "Fail to retrieve messages for {0!r}, caused by {1!r}".format(name, e)
            )

    def get_cid(self):
        root = ET.fromstring(self.xml)
        for i, drug in enumerate(root):
            cid = drug.find("./*[@Name='CID']")
            return cid.text

    def get_smiles(self):
        root = ET.fromstring(self.xml)
        for i, drug in enumerate(root):
            smiles = drug.find("./*[@Name='IsomericSmiles']")
            return smiles.text
