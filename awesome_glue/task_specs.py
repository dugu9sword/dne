# yapf:disable
TASK_SPECS = {
    "SST": {
        "path": "datasets/glue_data/SST-2/",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
        "banned_words": "banned_words/sentiment.txt",
        "num_labels": 2,  # if skip_label_indexing, this key should be specified
    },
    "AGNEWS": {
        "path": "datasets/agnews/",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
        "num_labels": 4,
    },
    "IMDB": {
        "path": "datasets/imdb/",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
        "banned_words": "banned_words/sentiment.txt",
        "num_labels": 2,
    },
    "RTE": {
        "path": "datasets/glue_data/RTE/",
        "sent1_col": "sentence1",
        "sent2_col": "sentence2",
        "label_col": "label",
        "skip_label_indexing": False,
    },
    "CoLA": {
        "path": "datasets/glue_data/CoLA/transformed",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
    },
    "QNLI": {
        "path": "datasets/glue_data/QNLI",
        "sent1_col": "question",
        "sent2_col": "sentence",
        "label_col": "label",
        "skip_label_indexing": False,
    }
}
# yapf:enable
