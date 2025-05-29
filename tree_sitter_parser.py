from tree_sitter import Language, Parser

# Build the language library for R
Language.build_library(
    'build/lang.so',
    [
        './tree-sitter-r'
    ]
)

LANGUAGE = Language('build/lang.so', 'r')

# Query for function definitions in R
FUNCTION_QUERY = LANGUAGE.query("""
(
  (binary_operator
    lhs: (identifier) @function_name
    operator: [ "<-" "=" ]
    rhs: (function_definition)
  )
)
""")

# Create a global parser instance for efficiency
global_parser = Parser()
global_parser.set_language(LANGUAGE)

def get_fn_name(code, parser=global_parser):
    """
    Extract the function name from an R function definition.
    """
    src = bytes(code, "utf8")
    tree = parser.parse(src)
    node = tree.root_node
    for cap, typ in FUNCTION_QUERY.captures(node):
        if typ == "function_name":
            return node_to_string(src, cap)
    return None




def node_to_string(src: bytes, node):
    """
    Convert a tree-sitter node to a string.
    """
    if isinstance(src, bytes):
        return src[node.start_byte:node.end_byte].decode("utf-8")
        # If src is already a string, just slice it
    elif isinstance(src, str):
        return src[node.start_byte:node.end_byte]
    else:
        raise TypeError("src must be either bytes or string")




def make_parser():
    """
    Create a new parser for R.
    """
    _parser = Parser()
    _parser.set_language(LANGUAGE)
    return _parser

# Query for return statements in R
RETURN_QUERY = LANGUAGE.query("""
(
  function_definition
    body: (braced_expression
      (call
        function: (return) @return
      )
    )
) @function_with_return
""")



def does_have_return(src, parser=global_parser):
    """
    Check if the given R code has a return statement.
    """
    tree = parser.parse(bytes(src, "utf8"))
    root = tree.root_node
    captures = RETURN_QUERY.captures(root)
    for node, _ in captures:
        # In R, return statements are often in the form of return(value)
        if len(node.children) <= 1: # includes "return" itself
            continue
        else:
            return True
    return False

if __name__ == "__main__":
    # Example R code to test
    code = """
my_function <- function(x, y) {
    z <- x + y
    return(z)
}
"""
    print("Function Name:", get_fn_name(code))
    print("Has Return Statement:", does_have_return(code))

    print(global_parser.parse(bytes(code, "utf8")).root_node.sexp())