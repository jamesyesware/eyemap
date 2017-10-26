import sys
import json

import pprint as pp

class EmailParser:

    def __init__(self):
        self.MAX_CHAR_COUNT = 60
        self.AUTO_RECORDS = 0 # machine generated
        self.GENUINE_RECORDS = 0 # human generated

    # Samples wil refer to how many items for each classification we want.
    def fetch_data(self, fpath, samples=200):
        with open(fpath, encoding='utf8') as f:
            content = f.readlines()
            training_data = []

            for el in content:
                if self.AUTO_RECORDS == samples and self.GENUINE_RECORDS == samples:
                    break

                sample = self.extract_auto_reply_and_subject(el, samples)

                if sample:
                    training_data.append(sample)

            return training_data

    def extract_auto_reply_and_subject(self, el, samples):
        el = json.loads(el)
        subject = self.pad_and_encode(el.get("subject"))
        auto_reply = el.get("auto_reply")

        if auto_reply and self.AUTO_RECORDS < samples:
            self.AUTO_RECORDS = self.AUTO_RECORDS + 1
            return {
                "subject": subject.decode("ascii") if subject is not None else "",
                "auto_reply": auto_reply
            }

        elif not auto_reply and self.GENUINE_RECORDS < samples:
            self.GENUINE_RECORDS = self.GENUINE_RECORDS + 1
            return {
                "subject": subject.decode("ascii") if subject is not None else "",
                "auto_reply": auto_reply
            }


    def pad_and_encode(self, subject):
        if subject is None:
            return

        char_count = len(subject)

        # trims or pads sentence to MAX_CHAR_COUNT
        if char_count > self.MAX_CHAR_COUNT:
            return subject[:self.MAX_CHAR_COUNT].encode('ascii','ignore')
        else:
            return subject.ljust(self.MAX_CHAR_COUNT).encode('ascii','ignore')

