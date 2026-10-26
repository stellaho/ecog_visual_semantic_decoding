LABEL_TO_INT = {
    "human face": 0, 
    "texts": 1, 
    "landscape": 2,
}

INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

FOLDER_RANGES = {
    "Trn01": (1, 600),
    "Trn02": (601, 1200),
    "Trn03": (1201, 1800),
    "Trn04": (1801, 2400),
    "Trn05": (2401, 3000),
    "Trn06": (3001, 3600),
}
