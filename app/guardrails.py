import re
INJECT = re.compile(r"ignore (all |any |previous |prior )*instructions|system prompt|reveal .*(key|secret|prompt)", re.I)
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{8,}\d")
MAX_LEN = 500

def input_ok(text):
    return len(text) <= MAX_LEN and not INJECT.search(text)

def redact(text):
    return PII.sub("[REDACTED]", text)
