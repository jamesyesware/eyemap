import json

def fetch_data(fpath, count=None):
    data = {
        "subject": [],
        "auto_reply": [],
        "auto_reply_count_true": 0,
        "auto_reply_count_false": 0
    }

    with open(fpath) as f:
        content = f.readlines()

    [extract_auto_reply_and_subject(data, el, count) for el in content]

    return data

def extract_auto_reply_and_subject(data, el, count=None):
    el = json.loads(el) 
    
    subject = el.get("subject")
    auto_reply = el.get("auto_reply")

    if subject is not None and auto_reply is not None:
        if count is not None:
            if auto_reply and data["auto_reply_count_true"] < count:
                data["subject"].append(subject)
                data["auto_reply"].append(1 if auto_reply else 0)
                data["auto_reply_count_true"] += 1
            if not auto_reply and data["auto_reply_count_false"] < count:
                data["subject"].append(subject)
                data["auto_reply"].append(1 if auto_reply else 0)
                data["auto_reply_count_false"] += 1
        else:
            data["subject"].append(subject)
            data["auto_reply"].append(1 if auto_reply else 0)
