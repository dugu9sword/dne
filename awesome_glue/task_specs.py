DEFAULT_TASK_PATH = 'glue_data'

#yapf:disable
TASK_SPECS = {
    "SST": {
        "path": f"{DEFAULT_TASK_PATH}/SST-2/",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
    },
    "RTE": {
        "path": f"{DEFAULT_TASK_PATH}/RTE/",
        "sent1_col": "sentence1",
        "sent2_col": "sentence2",
        "label_col": "label",
        "skip_label_indexing": False,
    },
    "CoLA": {
        "path": f"{DEFAULT_TASK_PATH}/CoLA/transformed",
        "sent1_col": "sentence",
        "sent2_col": None,
        "label_col": "label",
        "skip_label_indexing": True,
    }
}
#yapf:enable
