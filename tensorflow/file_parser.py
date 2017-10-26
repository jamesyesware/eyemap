import json

def fetch_data(fpath):
    data = {
        "subject": [],
        "auto_reply": []
    }

    with open(fpath) as f:
        content = f.readlines()

    [extract_auto_reply_and_subject(data, el) for el in content]

    return data

def extract_auto_reply_and_subject(data, el):
    el = json.loads(el) 
    
    subject = el.get("subject")
    auto_reply = el.get("auto_reply")

    if subject is not None and auto_reply is not None:
        data["subject"].append(subject)
        data["auto_reply"].append(1 if auto_reply else 0)
