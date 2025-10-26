from typing import Any, Dict, Sequence, List, Tuple, Union

"""
Python port of MATLAB hlp_varargin2struct(args, varargin).
"""

MANDATORY_SENTINEL = "__arg_mandatory__"
Name = Union[str, Sequence[str]]
PairList = List[Any]

def hlp_varargin2struct(
    args: Union[Sequence[Any], Dict[str, Any]],
    *defaults: Any,
) -> Dict[str, Any]:
    """
    Convert arguments and defaults into a structured dictionary.

    Parameters:
    args : Union[Sequence[Any], Dict[str, Any]]
        Input arguments as a sequence or dictionary.
    defaults : Any
        Default values for the arguments.

    Returns:
    Dict[str, Any]
        A dictionary containing the structured arguments.
    """
    # Normalize inputs and splice dicts that appear at "name" positions.
    if isinstance(args, dict):
        args_list: PairList = _flatten_structs_in_list([args])
    else:
        args_list = _flatten_structs_in_list(list(args))
    defaults_list: PairList = _flatten_structs_in_list(list(defaults)) if defaults else []

    # Parse defaults to gather (name, value) pairs and alternative name map.
    alt_to_canon: Dict[str, str] = {}
    def_pairs: List[Tuple[str, Any]] = []
    dpairs = _pairwise(defaults_list, what="Defaults") if defaults_list else []
    for nm, vv in dpairs:
        name: Name = nm
        # Handle alternative names: [canonical, alt1, alt2, ...]
        if isinstance(name, (list, tuple)):
            if not name:
                raise ValueError("Empty list of alternative names.")
            canonical = name[0]
            if not isinstance(canonical, str):
                raise ValueError("Canonical name must be a string.")
            for alt in name[1:]:
                if not isinstance(alt, str):
                    raise ValueError("Alternative names must be strings.")
                alt_to_canon[alt] = canonical
            name = canonical
        if not isinstance(name, str):
            raise ValueError("Default parameter name must be string or list/tuple of strings.")
        def_pairs.append((name, vv))

    # Check if defaults contain dotted names.
    any_dot_in_defaults = any("." in n for n, _ in def_pairs)

    # Build result dictionary from defaults.
    res: Dict[str, Any] = {}
    if not any_dot_in_defaults:
        # Handle non-dotted defaults: keep only the last assignment per name.
        last_values: Dict[str, Any] = {}
        for n, v in def_pairs:
            last_values[n] = v
        for n in sorted(last_values.keys()):
            res[n] = last_values[n]
    else:
        # Handle dotted defaults: assign sequentially.
        for n, v in def_pairs:
            _assign(res, n, v)

    # Apply overrides (args), remapping alternative names to canonical names.
    apairs = _pairwise(args_list, what="Arguments") if args_list else []
    for idx, (nm, vv) in enumerate(apairs):
        if not isinstance(nm, str):
            pos = 2 * idx + 1  # 1-based position for error message.
            raise ValueError(f"Invalid field name specified in arguments at position {pos}")
        name = alt_to_canon.get(nm, nm)
        _assign(res, name, vv)

    # Check for mandatory sentinel values (top-level only).
    top_values = list(res.values())
    missing_mask = [v == MANDATORY_SENTINEL for v in top_values]
    if any(missing_mask):
        fns = list(res.keys())
        missing = [fn for fn, miss in zip(fns, missing_mask) if miss]
        if len(missing) == 1:
            raise ValueError(f"The parameter {{{missing[0]}}} was unspecified but is mandatory.")
        else:
            all_but_last = ", ".join(missing[:-1])
            raise ValueError(f"The parameters {{{all_but_last}, {missing[-1]}}} were unspecified but are mandatory.")

    return res

def _flatten_structs_in_list(lst: Sequence[Any]) -> PairList:
    """
    Expand any dictionary at a name position into name/value pairs.
    Leave dictionaries that appear as values untouched.

    Parameters:
    lst : Sequence[Any]
        Input list containing potential dictionaries.

    Returns:
    PairList
        Flattened list with name/value pairs.
    """
    out: List[Any] = []
    k = 0
    L = list(lst)
    while k < len(L):
        item = L[k]
        if isinstance(item, dict):
            # Expand dictionary into name/value pairs.
            for key, val in item.items():
                out.append(key)
                out.append(val)
            k += 1
        else:
            # Copy the name and its value (if present).
            out.append(item)
            if k + 1 < len(L):
                out.append(L[k + 1])
            k += 2
    return out

def _set_nested(d: Dict[str, Any], dotted: str, value: Any) -> None:
    """
    Assign a dotted key (e.g., 'a.b.c') into nested dictionaries.

    Parameters:
    d : Dict[str, Any]
        Target dictionary.
    dotted : str
        Dotted key representing the nested structure.
    value : Any
        Value to assign.
    """
    parts = dotted.split(".")
    cur = d
    for i, p in enumerate(parts[:-1]):
        if p in cur and not isinstance(cur[p], dict):
            raise ValueError(
                f"Invalid field name specified in defaults/arguments: "
                f"'{'.'.join(parts[:i+1])}' is not a dictionary"
            )
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _assign(d: Dict[str, Any], name: str, value: Any) -> None:
    """
    Assign name/value into dictionary, supporting dotted names.

    Parameters:
    d : Dict[str, Any]
        Target dictionary.
    name : str
        Key name (supports dotted keys).
    value : Any
        Value to assign.
    """
    if "." in name:
        _set_nested(d, name, value)
    else:
        d[name] = value

def _pairwise(lst: Sequence[Any], *, what: str) -> List[Tuple[Any, Any]]:
    """
    Convert a flat list into [(name, value), ...].

    Parameters:
    lst : Sequence[Any]
        Input list to pair.
    what : str
        Description for error messages (e.g., 'Arguments' or 'Defaults').

    Returns:
    List[Tuple[Any, Any]]
        List of paired name/value tuples.
    """
    if len(lst) % 2 != 0:
        raise ValueError(f"{what} must be name/value pairs.")
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]
