
import signal
import re
from typing import Set
from tree_sitter_parser import LANGUAGE, global_parser

# Define a query to capture function calls in R
FUNCTION_CALL_QUERY = LANGUAGE.query("""
(call
  function: (identifier) @function_name
) @call
""")

# Define a query to capture function definitions in R
FUNCTION_DEF_QUERY = LANGUAGE.query("""
(
  (binary_operator
    lhs: (identifier) @function_name
    operator: [ "<-" "=" ]
    rhs: (function_definition)
  )
)
""")

# Mapping of common R functions to their packages
R_FUNCTION_TO_PACKAGE = {
    "ggplot": "ggplot2",
    "aes": "ggplot2",
    "geom_point": "ggplot2",
    "geom_line": "ggplot2",
    "geom_bar": "ggplot2",
    "geom_histogram": "ggplot2",
    "facet_wrap": "ggplot2",
    "facet_grid": "ggplot2",
    "theme_minimal": "ggplot2",
    "labs": "ggplot2",
    "scale_x_continuous": "ggplot2",
    "scale_y_continuous": "ggplot2",
    "mutate": "dplyr",
    "filter": "dplyr",
    "summarise": "dplyr",
    "summarize": "dplyr",
    "group_by": "dplyr",
    "arrange": "dplyr",
    "select": "dplyr",
    "rename": "dplyr",
    "distinct": "dplyr",
    "pivot_longer": "tidyr",
    "pivot_wider": "tidyr",
    "drop_na": "tidyr",
    "separate": "tidyr",
    "unite": "tidyr",
    "str_replace": "stringr",
    "str_detect": "stringr",
    "str_extract": "stringr",
    "str_split": "stringr",
    "str_to_lower": "stringr",
    "str_to_upper": "stringr",
    "map": "purrr",
    "map_df": "purrr",
    "walk": "purrr",
    "reduce": "purrr",
    "keep": "purrr",
    "discard": "purrr",
    "t.test": "stats",
    "chisq.test": "stats",
    "aov": "stats",
    "glm": "stats",
    "shapiro.test": "stats",
    "read_csv": "readr",
    "write_csv": "readr",
    "read_delim": "readr",
    "tibble": "tibble",
    "as_tibble": "tibble",
    "%>%": "dplyr"
}

# Common base R functions (no package import needed)
BASE_R_FUNCTIONS = {
    "data.frame", "mean", "sum", "length", "c", "list", "matrix", "vector", "print", "return",
    "cbind", "rbind", "t", "apply", "lapply", "sapply", "tapply", "mapply", "aggregate",
    "plot", "points", "lines", "abline", "hist", "boxplot", "barplot", "pie", "par",
    "head", "tail", "str", "summary", "dim", "nrow", "ncol", "colnames", "rownames",
    "subset", "merge", "order", "sort", "unique", "duplicated", "table", "prop.table",
    "as.numeric", "as.character", "as.factor", "as.data.frame", "as.matrix", "as.list",
    "ifelse", "paste", "paste0", "grep", "gsub", "sub", "strsplit", "tolower", "toupper",
    "lm", "anova", "predict", "coef", "residuals", "fitted", "sd", "var", "cor", "cov",
    "min", "max", "range", "quantile", "median", "abs", "sqrt", "log", "exp", "round", "ceiling", "floor"
}

def autoimports_r(code: str) -> str:
    """
    Infer missing library imports for R code by analyzing function calls and identify unknown functions.
    Excludes base R and user-defined functions from unknown function reporting.
    """
    try:
        def handler(signum, frame):
            raise Exception("Timeout")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)

        # Parse the code with tree-sitter
        tree = global_parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        # Get function definitions to exclude user-defined functions
        user_defined_functions: Set[str] = set()
        def_captures = FUNCTION_DEF_QUERY.captures(root)
        for node, tag in def_captures:
            if tag == "function_name":
                func_name = node.text.decode("utf8")
                user_defined_functions.add(func_name)

        # Get function calls
        used_functions: Set[str] = set()
        unknown_functions: Set[str] = set()
        call_captures = FUNCTION_CALL_QUERY.captures(root)
        for node, tag in call_captures:
            if tag == "function_name":
                func_name = node.text.decode("utf8")
                used_functions.add(func_name)
                # Skip base R and user-defined functions
                if func_name in BASE_R_FUNCTIONS or func_name in user_defined_functions:
                    continue
                # Flag as unknown if not in package mapping
                if func_name not in R_FUNCTION_TO_PACKAGE:
                    unknown_functions.add(func_name)

        # Check for existing library/require calls
        existing_imports: Set[str] = set()
        library_pattern = r"(library|require)\s*\(\s*['\"]?(\w+)['\"]?\s*\)"
        for match in re.finditer(library_pattern, code):
            existing_imports.add(match.group(2))

        # Determine required packages
        required_packages: Set[str] = set()
        for func in used_functions:
            if func in R_FUNCTION_TO_PACKAGE:
                package = R_FUNCTION_TO_PACKAGE[func]
                if package not in existing_imports:
                    required_packages.add(package)

        # Report unknown functions (if any)
        if unknown_functions:
            print(f"Unknown functions detected (not in known packages or defined within the function): {', '.join(sorted(unknown_functions))}")
            return None

        # Add missing library calls at the top of the code
        if required_packages:
            library_statements = "\n".join(f"library({pkg})" for pkg in sorted(required_packages))
            inferred_code = f"{library_statements}\n\n{code}"
        else:
            inferred_code = code

        signal.alarm(0)
        return inferred_code

    except Exception as e:
        signal.alarm(0)
        print(f"Error while inferring imports: {e}")
        return code


if __name__ == "__main__":

    input_code = """
    df <- data.frame(a = 1:5, b = 6:10)
    df <- df %>% mutate(c = a + b)  
    ggplot(df, aes(x = a, y = b)) + geom_point()  
    """


    r_code = """  
    my_function <- function(df) {
        df <- df %>% mutate(c = a + b)
        ggplot(df, aes(x = a, y = b)) + geom_point()
        return(df)
    }
    
    df <- data.frame(a = 1:5, b = 6:10)
    df <- my_function(df)
    df2 <- data.frame(a = 1:5, b = 6:10)
    df2 <- df %>% mutate(c = a + b)
    ggplot(df, aes(x = a, y = b)) + geom_point()
    """

    rcdose2 = """
    df <- data.frame(a = 1:5, b = 6:10)
    unknown_func(df) 
    """

    print(autoimports_r(input_code))
    print(autoimports_r(r_code))
    print(autoimports_r(rcdose2))