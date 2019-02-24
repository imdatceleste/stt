# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
"""
Character set module: This module is used to create two dictionaries that represent 
character set and its integer code definition and vice versa.

Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
All rights reserved.

Created: 2018-02-24 11:00 CET, ISO
"""

char_list = {
        'de_DE': u"' abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789",
        'en_US': u"' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        'tr_TR': u"' abcdefghijklmnopqrstuvwxyzäöüçşğıABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜÇŞĞİ0123456789"
        }

def _get_char2id_map(language):
    lang_chars = char_list.get(language, None)
    if lang_chars is None:
        return None
    cl = list(lang_chars)
    char_to_id = {s:i for i, s in enumerate(cl)}
    return char_to_id


def _get_id2char_map(language):
    lang_chars = char_list.get(language, None)
    if lang_chars is None:
        return None
    cl = list(lang_chars)
    id_to_char = {i:s for i, s in enumerate(cl)}
    return id_to_char


def get_language_chars(language_code):
    char_map = _get_char2id_map(language_code)
    index_map = _get_id2char_map(language_code)
    return char_map, index_map, len(char_map) + 1


def check_language_code(language_code):
    return language_code in char_list.keys()


def clean_text(text, char_map):
    out_text = ''
    allowed_chars = char_map.keys()
    for ch in text:
        if ch in allowed_chars:
            out_text += ch
    return out_text

