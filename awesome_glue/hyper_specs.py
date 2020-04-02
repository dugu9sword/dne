#yapf: disable
HYPER_SPECS = {
    "*|*": {
        "num_epochs": 15,
        "batch_size": 32
    },
    "*|BERT": {
        "num_epochs": 3,
        "batch_size": 16
    },
    "SNLI|*": {
        "batch_size": 128
    },
    "SST|BERT": {
        "batch_size": 32
    },
    "SNLI|BERT": {
        "batch_size": 32
    }
}
#yapf: enable


def get_hyper(task_id, arch_id, key):
    ta_keys = [f"{task_id}|{arch_id}", 
               f"{task_id}|*", 
               f"*|{arch_id}",
               "*|*"]
    ret = [None, None, None, None]
    for ta_id, ta_key in enumerate(ta_keys):
        if ta_key in HYPER_SPECS:
            if key in HYPER_SPECS[ta_key]:
                ret[ta_id] = HYPER_SPECS[ta_key][key]
    if ret[0]:
        return ret[0]
    if ret[1] is not None and ret[2] is not None:
        raise Exception('ambiguty')
    if ret[1] is not None:
        return ret[1]
    if ret[2] is not None:
        return ret[2]
    if ret[3] is not None:
        return ret[3]
    else:
        raise Exception('Unknown')
