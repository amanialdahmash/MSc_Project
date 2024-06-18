import re
from typing import List

from spec_repair.enums import Learning


def split_expression_to_raw_components(exp: str) -> List[str]:
    exp_components: List[str] = exp.split("->")
    if len(exp_components) == 1:
        exp = re.sub(r"(G|GF)\(\s*", r"\1(true -> ", exp_components[0])
        exp_components = exp.split("->")
    exp_components = [comp.strip() for comp in exp_components]
    return exp_components


def eventualise_consequent(exp, learning_type: Learning):
    match learning_type:
        case Learning.ASSUMPTION_WEAKENING:
            line = split_expression_to_raw_components(exp)
            return eventualise_consequent_assumption(line)
        case Learning.GUARANTEE_WEAKENING:
            line = split_expression_to_raw_components(exp)
            return eventualise_consequent_assumption(line)
            raise NotImplemented(
                "Not sure yet if we want to weaken guarantees by introducing eventually to their consequent.")
        case _:
            raise ValueError("No such learning type")


def extract_contents_of_temporal(expression: str):
    # Remove "next", "prev", or "X" (case-insensitive) and surrounding parentheses
    return re.sub(r'(?i)(next|prev|X)\s*\(([^)]*)\)|\)$', r'\2', expression)


def eventualise_consequent_assumption(line: List[str]):
    antecedent = line[0]
    consequent = line[1]
    consequent_without_temporal = extract_contents_of_temporal(consequent)
    ev_consequent = re.sub(r'^(.*?)(;)?$', r'F(\1)\2', consequent_without_temporal)
    output = antecedent + "->" + ev_consequent
    return '\t' + output + "\n"
