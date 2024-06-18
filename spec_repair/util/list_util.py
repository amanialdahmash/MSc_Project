def re_line_spec(spec: list[str]) -> list[str]:
    """
    Move multiple newlines to new elems in list.
    Ensures every separate elem has a newline at the end.
    e.g.: ["Anna\n\n", "\n", "eats\n", "potatoes"]
        -> ["Anna\n", "\n", "\n", "eats\n", "potatoes\n"]
    :param spec: Specification as list of strings
    :return: new_spec: Specification reformatted as explained above
    """
    return [line + '\n' for line in ''.join(spec).split("\n")]
