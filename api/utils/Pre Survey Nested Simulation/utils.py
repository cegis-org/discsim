def flatten_dict(nested_dict):
    items = []
    for value in nested_dict.values():
        if isinstance(value, dict):
            items.extend(flatten_dict(value))
        else:
            items.append(value)
    return items