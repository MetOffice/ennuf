#  (C) Crown Copyright, Met Office, 2023.
import re


def split_except_in_single_quotes(string: str):
    """
    Returns string split as if string.split() was called but ignoring parts of the string surrounded in single quotes,
    which are kept together. For example, "Here is an example with 'quoted text' in it" would return
    ['here', 'is', 'an', 'example', 'with', "'quoted text'", 'in', 'it']
    """
    # split into non-quoted and quoted regions
    substrings = re.findall(r"([^']*)|('[^']*')", string)
    result = []
    for non_quoted_substr, quoted_substr in substrings:
        if non_quoted_substr:
            for part in non_quoted_substr.split():
                result.append(part)
        if quoted_substr:
            result.append(quoted_substr)
    return result
