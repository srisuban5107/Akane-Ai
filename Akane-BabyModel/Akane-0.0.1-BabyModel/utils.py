def clean_whatsapp_line(line):
    import re
    return re.sub(r"\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2} - .*?: ", "", line)
