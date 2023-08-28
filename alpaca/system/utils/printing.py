def to_string(obj: object, indentation: str = "  ") -> str:
    out = type(obj).__name__ + " {\n"
    for k, v in obj.__dict__.items():
        out += f"{indentation}{k}: {v}\n"
    out += "}"
    return out


def add_indentation(s: str, indentation: str = "  ") -> str:
    return s.replace("\n", "\n" + indentation)
