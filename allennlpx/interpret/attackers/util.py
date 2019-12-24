def select(ordered_idxs=[5, 2, 1, 3, 0, 4], num_to_select=3, selected={2, 3, 4}, max_num=5):
    """
    selected = set()
    new_select = elder_select([5, 2, 1, 3, 0, 4], selected=selected)
    assert(new_select == [5, 2, 1])
    selected.update(set(new_select))
    new_select = elder_select([5, 2, 0, 3, 1, 4], selected=selected)
    assert(new_select == [5, 2, 0])
    selected.update(set(new_select))
    new_select = elder_select([3, 4, 0, 1, 2, 5], selected=selected)
    assert(new_select == [3, 0, 1])
    """
    ret = []
    total_num = len(selected)
    for ele in ordered_idxs:
        if len(ret) == num_to_select:
            break
        if ele in selected:
            ret.append(ele)
        else:
            if total_num == max_num:
                continue
            else:
                ret.append(ele)
                total_num += 1
    return ret