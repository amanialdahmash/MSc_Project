# TODO: consider between:
# * Extension of spot.formula/similar class
# * Complete OOP redesign
import re
from collections import OrderedDict
from typing import List, Set

import pandas as pd

from spec_repair.enums import ExpType, When
from spec_repair.old.patterns import PRS_REG


class LTLFormula:
    formula: str
    # >>>>> NOT YET IN USE >>>>>
    type: ExpType
    name: str
    antecedent: set[str]
    consequent: set[str]
    when: When
    # <<<<< NOT YET IN USE <<<<<

    def __init__(self, formula: str):
        if not isinstance(formula, str) or '\n' in formula:
            raise ValueError("Formula must be a one-line string.")
        self.formula = formula

    def __getattr__(self, name):
        # If the attribute is a string method, apply it to the stored formula
        if hasattr(self.formula, name) and callable(getattr(self.formula, name)):
            return getattr(self.formula, name)
        raise AttributeError(f"'LTLFormula' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Make the class immutable by preventing attribute changes after initialization
        if hasattr(self, "_data"):
            raise AttributeError("Cannot modify attributes of 'LTLFormula' object.")
        super().__setattr__(name, value)


class Assumption(LTLFormula):
    pass


class Guarantee(LTLFormula):
    pass


class Trace:
    variables: Set[str]
    path: List[Set[str]]

    def __init__(self, file_name: str):
        raise NotImplemented

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.path):
            raise StopIteration
        value = self.path[self.index]
        self.index += 1
        return value


# Type renames for consistency
Trace = str
CounterStrategy = List[str]
Spec = pd.DataFrame


def log_to_asp_trace(lines: str, trace_name: str = "trace_name_0") -> str:
    """
    Converts a runtime log into a workable trace string
    i.e.
    ->
    :param lines: Lines from log file
    :param trace_name: Name of Log
    :return: Trace string
    """
    ret = ""
    for i, line in enumerate(lines.split("\n")):
        ret += log_line_to_asp_trace(line, i, trace_name)
        ret += "\n"
    return ret


# TODO: figure out what to do about PREV_aux_0 or Zn
def log_line_to_asp_trace(line: str, idx: int = 0, trace_name: str = "trace_name_0") -> str:
    """
    Converts one line from a runtime log into a workable trace string
    i.e.
    ->
    :param line: <highwater:false, methane:false, pump:false, PREV_aux_0:false, Zn:0>
    :param idx: index where log line resides
    :param trace_name:
    :return:     not_holds_at(current,highwater,idx,trace_name).
                 not_holds_at(current,methane,idx,trace_name).
                 not_holds_at(current,pump,idx0,trace_name).
    """
    pairs = extract_string_boolean_pairs(line)
    filtered_pairs = [(key, value == 'true') for key, value in pairs if not key.startswith(('PREV', 'NEXT', 'Zn'))]
    ret = ""
    for env_var, is_true in filtered_pairs:
        ret += f"{'' if is_true else 'not_'}holds_at(current,{env_var},{idx},{trace_name}).\n"

    return ret


def extract_string_boolean_pairs(line):
    """
    Get all pairs of strings and booleans of form 'name:val'
    :param line:
    :return:
    """
    pattern = r"\b([a-zA-Z_][\w]*):(\btrue\b|\bfalse\b)"
    pairs = re.findall(pattern, line)
    return pairs


def spectra_to_df(spec: List[str]) -> pd.DataFrame:
    """
    Converts formatted Spectra file into Pandas DataFrame for manipulation into ASP.

    :param spec: Spectra specification as List of Strings.
    :return: Pandas DataFrame containing GR(1) expressions converted into antecedent/consequent.
    """
    formula_list = []
    for i, line in enumerate(spec):
        words = line.split(" ")
        if line.find('--') >= 0:
            name = re.sub(r":|\s", "", words[2])
            formula = re.sub('\s*', '', spec[i + 1])
            ant_list = []
            cons_list = []

            pRespondsToS, when = gr1_type_of(formula)

            ant_when = when
            if pRespondsToS:
                ant_when = When.ALWAYS

            formula_parts = formula.replace(");", "").split("->")
            if len(formula_parts) == 1:
                antecedent = ""
                consequent = re.sub(r"[^\(]*\(", "", formula_parts[0], 1)
            else:
                antecedent = re.sub(r"[^\(]*\(", "", formula_parts[0], 1)
                consequent = formula_parts[1]
            if pRespondsToS:
                consequent = re.sub(r"^F\(", "", consequent)
            consequent_disjuncts = consequent.split("|")
            antecedent_disjuncts = antecedent.split("|")
            for antecedent in antecedent_disjuncts:
                if antecedent != "":
                    ant_list.append(convert_assignments(antecedent, ant_when, consequent=False))
            for consequent in consequent_disjuncts:
                cons_list.append(convert_assignments(consequent, when, consequent=True))

            formula_list.append(
                [words[0], name, formula,
                 ant_list,
                 cons_list, when]
            )
    columns_and_types = OrderedDict([
        ('type', str),
        ('name', str),
        ('formula', str),
        ('antecedent', object),  # list[str]
        ('consequent', object),  # list[str]
        ('when', object)  # When
    ])
    spec_df = pd.DataFrame(formula_list, columns=list(columns_and_types.keys()))
    # Set the data types for each column
    for col, dtype in columns_and_types.items():
        spec_df[col] = spec_df[col].astype(dtype)

    return spec_df


def gr1_type_of(formula):
    '''
    :param formula:
    :return: pRespondsToS, when
    '''
    formula = re.sub('\s*', '', formula)
    eventually = re.search(r"^GF", formula)
    pRespondsToS = PRS_REG.search(formula)
    initially = not re.search(r"^G", formula)
    if eventually:
        when = When.EVENTUALLY
    elif initially:
        when = When.INITIALLY
    elif pRespondsToS:
        when = When.EVENTUALLY
    else:
        when = When.ALWAYS
    return pRespondsToS, when


def convert_assignments(expression, when, consequent):
    expression_list = []
    exp_parts = expression.split("&")
    for part in exp_parts:
        # TODO: assuming no nesting of next/prev
        temp_op = "current"
        for theta in ["next", "prev", "PREV"]:
            if theta in part:
                temp_op = theta.lower()
                part = part.replace(theta + "(", "")
        if when == When.EVENTUALLY:
            temp_op = "eventually"
        prefix = ""
        if part.find("=false") >= 0:
            prefix = "not_"
        atom = re.sub(r"\(|\)", "", part.split("=")[0])
        part_string = f"{prefix}holds_at({temp_op},{atom},T,S)"
        if when == When.INITIALLY:
            part_string = part_string.replace(",T,", ",0,")
        expression_list.append(part_string)
    trivial_exp = re.compile(r"^holds_at\((.*),true|^not_holds_at\((.*),false")
    expression_list = [x for x in expression_list if not trivial_exp.search(x)]
    output = ',\n\t'.join(expression_list)
    return output


def filter_expressions_of_type(formula_df: pd.DataFrame, expression: ExpType) -> pd.DataFrame:
    return formula_df.loc[formula_df['type'] == str(expression)]
