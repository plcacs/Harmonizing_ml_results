import ast
from typing import Set


def parse_type_to_ast(type_str: str) -> ast.expr:
    """Parse type string to AST expression"""
    try:
        # Handle Python 3.10+ union syntax
        if " | " in type_str:
            parts = [part.strip() for part in type_str.split(" | ")]
            if "None" in parts:
                parts.remove("None")
                if len(parts) == 1:
                    # Single type | None -> Union[type, None]
                    union_parts = f"{parts[0]}, None"
                    return ast.parse(f"Union[{union_parts}]", mode="eval").body
                else:
                    # Multiple types | None -> Union[types, None]
                    union_parts = ", ".join(parts) + ", None"
                    return ast.parse(f"Union[{union_parts}]", mode="eval").body
            else:
                # Multiple types -> Union[types]
                union_parts = ", ".join(parts)
                return ast.parse(f"Union[{union_parts}]", mode="eval").body

        # Handle Optional syntax - convert to Union
        if type_str.startswith("Optional["):
            inner = type_str[len("Optional[") : -1]
            return ast.parse(f"Union[{inner}, None]", mode="eval").body

        # Parse normally
        return ast.parse(type_str, mode="eval").body
    except Exception as e:
        print(f"Error parsing '{type_str}': {e}")
        # Fallback: return as string literal
        return ast.Constant(value=type_str)


def extract_type_components(expr: ast.expr) -> Set[str]:
    """Extract all type components from AST expression"""
    components = set()

    if isinstance(expr, ast.Name):
        components.add(expr.id)
    elif isinstance(expr, ast.Constant):
        if expr.value is None:
            components.add("None")
        else:
            components.add(str(expr.value))
    elif isinstance(expr, ast.Subscript):
        # Handle List[str], Dict[str, int], etc.
        if isinstance(expr.value, ast.Name):
            components.add(expr.value.id)
        if isinstance(expr.slice, ast.Tuple):
            for elt in expr.slice.elts:
                components.update(extract_type_components(elt))
        else:
            components.update(extract_type_components(expr.slice))
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        # Handle | operator
        components.update(extract_type_components(expr.left))
        components.update(extract_type_components(expr.right))
    elif isinstance(expr, ast.Call):
        # Handle Union[...], Optional[...], etc.
        if isinstance(expr.func, ast.Name) and expr.func.id in ["Union", "Optional"]:
            for arg in expr.args:
                components.update(extract_type_components(arg))

    return components


def normalize_type_components(components: Set[str]) -> Set[str]:
    """Normalize type components for comparison"""
    normalized = set()
    for comp in components:
        # Normalize built-in types
        if comp == "List":
            normalized.add("list")
        elif comp == "Dict":
            normalized.add("dict")
        elif comp == "Set":
            normalized.add("set")
        elif comp == "Tuple":
            normalized.add("tuple")
        elif comp == "Optional":
            # Optional is equivalent to Union, so we can ignore it
            continue
        else:
            normalized.add(comp)
    return normalized


def types_equivalent_semantic(type1: str, type2: str) -> bool:
    """Compare types semantically using AST parsing"""
    try:
        print(f"\nComparing: '{type1}' vs '{type2}'")

        # Parse both types to AST
        ast1 = parse_type_to_ast(type1)
        ast2 = parse_type_to_ast(type2)

        # Extract components
        components1 = extract_type_components(ast1)
        components2 = extract_type_components(ast2)

        print(f"Components1: {components1}")
        print(f"Components2: {components2}")

        # Normalize components
        norm1 = normalize_type_components(components1)
        norm2 = normalize_type_components(components2)

        print(f"Normalized1: {norm1}")
        print(f"Normalized2: {norm2}")

        # Check for exact equality first
        if norm1 == norm2:
            print("Exact match!")
            return True

        # Check for subtype relationships
        # Remove None from both sets for comparison
        set1_no_none = norm1 - {"None"}
        set2_no_none = norm2 - {"None"}

        print(f"Set1 without None: {set1_no_none}")
        print(f"Set2 without None: {set2_no_none}")

        # Check if one type is a subset of the other (subtype relationship)
        if set1_no_none == set2_no_none:
            print("Same base types!")
            return True
        elif set1_no_none.issubset(set2_no_none) or set2_no_none.issubset(set1_no_none):
            print("Subtype relationship!")
            return True

        print("No match")
        return False
    except Exception as e:
        print(f"Error in comparison: {e}")
        # Fallback to string comparison
        return type1.strip() == type2.strip()


# Test cases
test_cases = [
    ("Optional[Hash32]", "Hash32"),
    ("Hash32", "Optional[Hash32]"),
    ("str | None", "str"),
    ("str", "Optional[str]"),
    ("list[str] | None", "list[str]"),
]

for t1, t2 in test_cases:
    print("=" * 50)
    result = types_equivalent_semantic(t1, t2)
    print(f"Final result: {result}")
