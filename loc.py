import uuid


def is_cn():
    return hex(uuid.getnode())[2:].upper() == '43F648F4EEBB'