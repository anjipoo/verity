# ml/preprocessing/berka_schema.py
# Single source of truth for the Berka (PKDD'99) tables: dtypes, keys, privacy
# decisions, and baseline categorical domains. Nothing here is inferred by pandas.

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

# Column kinds understood by berka_ingest:
#   int / float : numeric tokens (optional placeholders listed in NUMERIC_PLACEHOLDERS)
#   text        : quoted text; unquoted-empty -> missing, "" and " " are kept as-is
#   id_str      : unquoted digit string kept as text (identifier, not a measure)
#   date        : YYMMDD
#   datetime    : YYMMDD HH:MM:SS
SCHEMA = {
    "account": {"account_id": "int", "district_id": "int", "frequency": "text", "date": "date"},
    "card": {"card_id": "int", "disp_id": "int", "type": "text", "issued": "datetime"},
    "client": {"client_id": "int", "birth_number": "text", "district_id": "int"},
    "disp": {"disp_id": "int", "client_id": "int", "account_id": "int", "type": "text"},
    # raw names A1..A16; renamed after loading with DISTRICT_COLUMN_NAMES
    "district": {
        "A1": "int", "A2": "text", "A3": "text", "A4": "int", "A5": "int",
        "A6": "int", "A7": "int", "A8": "int", "A9": "int", "A10": "float",
        "A11": "int", "A12": "float", "A13": "float", "A14": "int",
        "A15": "int", "A16": "int",
    },
    "loan": {"loan_id": "int", "account_id": "int", "date": "date", "amount": "int",
             "duration": "int", "payments": "float", "status": "text"},
    "order": {"order_id": "int", "account_id": "int", "bank_to": "text",
              "account_to": "id_str", "amount": "float", "k_symbol": "text"},
    "trans": {"trans_id": "int", "account_id": "int", "date": "date", "type": "text",
              "operation": "text", "amount": "float", "balance": "float",
              "k_symbol": "text", "bank": "text", "account": "id_str"},
}

# Unquoted "?" in these district columns is a missing-value marker (observed in the file)
NUMERIC_PLACEHOLDERS = {("district", "A12"): {"?"}, ("district", "A15"): {"?"}}

# Low-cardinality text columns stored as pandas category
CATEGORY_COLUMNS = {
    "account": ["frequency"], "card": ["type"], "disp": ["type"],
    "district": ["region"], "loan": ["status"], "order": ["bank_to", "k_symbol"],
    "trans": ["type", "operation", "k_symbol", "bank"],
}

# Privacy: removed at ingestion, never written to snapshots, reports, interim files or logs
DROPPED_COLUMNS = {"client": ["birth_number"]}

PRIMARY_KEYS = {"account": "account_id", "card": "card_id", "client": "client_id",
                "disp": "disp_id", "district": "district_id", "loan": "loan_id",
                "order": "order_id", "trans": "trans_id"}

# (child table, child column, parent table, parent column)
FOREIGN_KEYS = [
    ("account", "district_id", "district", "district_id"),
    ("client", "district_id", "district", "district_id"),
    ("disp", "client_id", "client", "client_id"),
    ("disp", "account_id", "account", "account_id"),
    ("card", "disp_id", "disp", "disp_id"),
    ("loan", "account_id", "account", "account_id"),
    ("order", "account_id", "account", "account_id"),
    ("trans", "account_id", "account", "account_id"),
]

# Baseline domains OBSERVED in the files on 2026-10-04 (not taken from documentation).
# They exist to detect change, not to assert meaning. Missing values are checked separately.
EXPECTED_DOMAINS = {
    ("account", "frequency"): {"POPLATEK MESICNE", "POPLATEK PO OBRATU", "POPLATEK TYDNE"},
    ("card", "type"): {"classic", "gold", "junior"},
    ("disp", "type"): {"DISPONENT", "OWNER"},
    ("loan", "status"): {"A", "B", "C", "D"},
    ("order", "k_symbol"): {" ", "LEASING", "POJISTNE", "SIPO", "UVER"},
    ("trans", "type"): {"PRIJEM", "VYBER", "VYDAJ"},
    ("trans", "operation"): {"PREVOD NA UCET", "PREVOD Z UCTU", "VKLAD", "VYBER", "VYBER KARTOU"},
    ("trans", "k_symbol"): {"", " ", "DUCHOD", "POJISTNE", "SANKC. UROK", "SIPO",
                            "SLUZBY", "UROK", "UVER"},
    ("trans", "bank"): {"", "AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP", "QR", "ST", "UV", "WX", "YZ"},
    ("order", "bank_to"): {"AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP", "QR", "ST", "UV", "WX", "YZ"},
}

# Observed dataset period; used only to flag dates outside it
DATE_WINDOW = ("1993-01-01", "1998-12-31")

# SHA-256 of the verified raw files (inspection of 2026-10-04)
EXPECTED_SHA256 = {
    "account": "58d7f50abd72e9b1a5568346f74bb54cd71224ee1db9f09a27d7cac563f38cc6",
    "card": "fc669bde6adf6457d87421c0bfb218e9c384a7032c6accd348d207a405e72109",
    "client": "e435c6b92d246f4f0dfd5e2827469d745c06238714c32b3ffb415eebe794e1a7",
    "disp": "ebd801f77b6d322e8ebc08e52f188e7c8fca539325f85f57f8c73434da9d32d8",
    "district": "7f03cf3b9b82f0fdcc3abdf6cc716f145db8e9875c68e2d2e2f7151e9ecf4df3",
    "loan": "68535f609a254aa7a3f03dd8e27dcb822b532df12a0d6046f0666b8dc0b8ae8e",
    "order": "035930fa6acd2ca42a935e654b21e1bb260248f49b6dc6e7de6351b7c4d56d02",
    "trans": "75ab2f39df9d79d79c5c900de90ddd28248b689f214598ac9fa2ff0f574a70d2",
}

# From the official "Guide to the Financial Data Set" (Financial Data description.pdf,
# SHA-256 67227f33f4fc6f8526ca5216fc6ac55ef86393b4f6b3cb6c69a0969c9ffa2b16, read 2026-10-05)
DISTRICT_COLUMN_NAMES = {
    "A1": "district_id", "A2": "district_name", "A3": "region", "A4": "n_inhabitants",
    "A5": "n_municipalities_lt_499", "A6": "n_municipalities_500_1999",
    "A7": "n_municipalities_2000_9999", "A8": "n_municipalities_gt_10000",
    "A9": "n_cities", "A10": "urban_inhabitants_ratio", "A11": "average_salary",
    "A12": "unemployment_rate_1995", "A13": "unemployment_rate_1996",
    "A14": "entrepreneurs_per_1000", "A15": "crimes_1995", "A16": "crimes_1996",
}

OFFICIAL_ROW_COUNTS = {"account": 4500, "client": 5369, "disp": 5369, "order": 6471,
                       "trans": 1056320, "loan": 682, "card": 892, "district": 77}

# Values the official description lists, per column (disp.type is described only as owner/user)
DOCUMENTED_VALUES = {
    ("account", "frequency"): {"POPLATEK MESICNE", "POPLATEK TYDNE", "POPLATEK PO OBRATU"},
    ("card", "type"): {"junior", "classic", "gold"},
    ("loan", "status"): {"A", "B", "C", "D"},
    ("order", "k_symbol"): {"POJISTNE", "SIPO", "LEASING", "UVER"},
    ("trans", "type"): {"PRIJEM", "VYDAJ"},
    ("trans", "operation"): {"VYBER KARTOU", "VKLAD", "PREVOD Z UCTU", "VYBER", "PREVOD NA UCET"},
    ("trans", "k_symbol"): {"POJISTNE", "SLUZBY", "UROK", "SANKC. UROK", "SIPO", "DUCHOD", "UVER"},
}
DOCUMENTATION = {
    "file": "Financial Data description.pdf",
    "url": "https://raw.githubusercontent.com/jlacko/berka-dataset/master/Financial%20Data%20description.pdf",
    "sha256": "67227f33f4fc6f8526ca5216fc6ac55ef86393b4f6b3cb6c69a0969c9ffa2b16",
}