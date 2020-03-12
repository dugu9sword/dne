import os

def use_fairseq_9():
    if os.path.exists("fairseq"):
        os.rename("fairseq", "fairseq0.6")

def use_fairseq_6():
    if os.path.exists("fairseq0.6"):
        os.rename("fairseq0.6", "fairseq")