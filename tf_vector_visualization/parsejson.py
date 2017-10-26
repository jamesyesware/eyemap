import sys
import json
import re

import pprint as pp

MAX_CHAR_COUNT = 60

def fetch_data(fpath):
    with open(fpath, encoding='utf8') as f:
        content = f.readlines()

    return [str(extract_auto_reply_and_subject(el))+", " for el in content]

def extract_auto_reply_and_subject(el):
    el = json.loads(el) 
    
    subject = el.get("subject")
    auto_reply = el.get("auto_reply")
    if auto_reply == True:
        for char in '?.!/}|/;:,-/{123456789]@)(0[':  
            subject = subject.replace(char,'') 

        return subject.lower()