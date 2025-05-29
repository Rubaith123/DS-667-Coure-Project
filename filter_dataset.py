import datasets
import os
from tree_sitter_parser import global_parser, LANGUAGE, does_have_return, make_parser
# import benchmark_data
from tqdm import tqdm
import torch
import argparse
from vllm import LLM, SamplingParams
import random
import re






# Tree-sitter query for R functions with roxygen2 docstrings
FN_BLOCK_QUERY = LANGUAGE.query("""
(
  (comment) @docstring
  .
  (binary_operator
    lhs: (identifier) @function_name
    operator: [ "<-" "=" ]
    rhs: (function_definition
      body: (braced_expression))
  ) @function.def
  (#match? @docstring "^#'")
)
""")


def r_extract_docstring(code):
    # Find roxygen2 comments before the function definition
    lines = code.splitlines()
    doc_lines = []
    code_lines = []
    in_doc = True

    for line in lines:
        stripped = line.strip()
        if in_doc and stripped.startswith("#'"):
            doc_lines.append(stripped[2:].strip())  # Remove #' prefix
        else:
            in_doc = False
            code_lines.append(line)

    doc = "\n".join(doc_lines).strip()
    code = "\n".join(code_lines).strip()
    return doc, code


def template_few_shot(code, answer, rationale):
    doc, code = r_extract_docstring(code)
    assert answer == "No" or answer == "Yes"  #ensures that the variable answer is either "No" or "Yes".
    prompt = f"""<issue_start>username_0: I have a function in R and I'd like someone to check my description of this function.
I'm doing this so that I can write a good docstring for this function.

Here is the code for the function:
```R
{code}
```

Here is my description of this program:
```
{doc}
```

Do not attempt to execute the function or to judge its correctness.
Answer with "Yes" or "No" depending on if my description has enough information alone to re-implement the function.
Also, answer with "No" if the description does not match the function.<issue_comment>username_1: Sure, no problem. I will be able to help.
My answer is: {answer}

{rationale}

Upvotes: 200"""
    return prompt

FEW_SHOTS = [
    (
        '''#' Generate a sequence of numbers
#' @param n Number of elements
#' @return A vector of numbers from 1 to n
sequence <- function(n) {
    return(1:n)
}''',
        "Yes",
        "The docstring clearly describes that the function generates a sequence of numbers from 1 to n, which matches the implementation using R's 1:n syntax."
    ),
    (
        '''#' Calculate the mean of a vector
#' @param x A numeric vector
mean_vector <- function(x) {
    sum_x <- sum(x)
    n <- length(x)
    return(sum_x / n)
}''',
        "Yes",
        "The docstring accurately states that the function calculates the mean of a numeric vector, and the implementation correctly computes sum(x)/length(x)."
    ),
    (
        '''#' Perform a linear regression
#' @param x Predictor variable
#' @param y Response variable
regression <- function(x, y) {
    model <- lm(y ~ x)
    return(summary(model))
}''',
        "No",
        "The docstring mentions performing a linear regression but does not specify that the function returns a summary of the model, which is a critical detail for re-implementation."
    ),
    (
        '''#' Filter positive values
#' @param x A vector
#' @return Positive values from x
filter_positive <- function(x) {
    return(x[x > 0])
}''',
        "Yes",
        "The docstring is concise and matches the implementation, which filters and returns positive values from the input vector."
    ),
    (
        '''#' Compute factorial
#' @param n An integer
factorial <- function(n) {
    if (n <= 1) return(1)
    return(n * factorial(n - 1))
}''',
        "Yes",
        "The docstring states the function computes a factorial, and the recursive implementation matches this description."
    ),
    (
        '''#' Data processing function
#' @param df A data frame
process_data <- function(df) {
    df$normalized <- df$value / max(df$value)
    df$category <- as.factor(df$category)
    return(df)
}''',
        "No",
        "The docstring is too vague, only mentioning 'data processing' without specifying that it normalizes a 'value' column and converts a 'category' column to a factor."
    ),
    (
        '''#' Sort a vector
#' @param x A vector
#' @return Sorted vector
sort_vector <- function(x) {
    return(sort(x, decreasing = TRUE))
}''',
        "No",
        "The docstring implies a standard sort but does not mention that the function sorts in descending order, which is a critical detail."
    )
]

def prompt_fmt(code):
    doc, code = r_extract_docstring(code)
    random.shuffle(FEW_SHOTS)
    buf = ""
    for few in FEW_SHOTS:
        buf += template_few_shot(*few)
    buf += f"""<issue_start>username_0: I have a function in R and I'd like someone to check my description of this function.
I'm doing this so that I can write a good docstring for this function.

Here is the code for the function:
```R
{code}
```

Here is my description of this program:
```
{doc}
```

Do not attempt to execute the function or to judge its correctness.
Answer with "Yes" or "No" depending on if my description has enough information alone to re-implement the function.
Also, answer with "No" if the description does not match the function.
Upvotes: 100<issue_comment>username_1: Sure, no problem. I will be able to help.
My answer is:"""
    return buf

def auto_dtype():
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "auto"

def chunkify(lst, n):
    chunks = []
    for i in range(0, len(lst), n):
        chunk = []
        for j in range(n):
            if i + j < len(lst):
                chunk.append(lst[i + j])
        chunks.append(chunk)
    return chunks


BAD_WORDS = ["todo", "fixme", "bug"]
BAD_IMPORTS = ["argparse", "sys", "os", "subprocess", "reticulate", "rJava"]
BAD_IMPORTS = [f"library({b})" for b in BAD_IMPORTS] + [f"require({b})" for b in BAD_IMPORTS]
BAD_SUBSTRINGS = BAD_WORDS + BAD_IMPORTS

# bench_filter = benchmark_data.filter_out()
# all_bench = bench_filter.get("r_benchmarks", [])  # Adjust for R benchmarks if available

def pre_filtering(ex):
    code = ex[args.content_col]
    code_bytes = code.encode('utf-8')

    # Filter out bad substrings
    lower = code.lower()
    for word in BAD_SUBSTRINGS:
        if word in lower:
            return False

    # for b in all_bench:
    #     if b in code:
    #         return False

    # Filter code with too many lines
    lines = code.split("\n")
    if len(lines) > 150:
        return False

    # Filter functions with no arguments
    for line in lines:
        if line.strip().startswith(('<', '=')) and 'function()' in line:
            return False

    # Filter out functions with no return statement
    parser = make_parser()
    if not does_have_return(code, parser=parser):
        return False

    try:
        tree = global_parser.parse(code_bytes)
        captures = FN_BLOCK_QUERY.captures(tree.root_node)
        docstring_nodes = [node for node, ty in captures if ty == "docstring"]
        function_node = next((node for node, ty in captures if ty == "function.def"), None)

        if not docstring_nodes or not function_node:
            return False

        # Verify roxygen2 docstring format
        for node in docstring_nodes:
            docstring_text = node.text.decode('utf-8')
            if not docstring_text.startswith("#'"):
                return False

    except Exception as e:
        print(f"Error in filtering: {e}")
        return False

    return True


def unindent(s):
    lines = s.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    min_indent = min(len(line) - len(line.lstrip())
                     for line in non_blank_lines) if non_blank_lines else 0
    unindented_lines = [line[min_indent:] if len(
        line) >= min_indent else line for line in lines]
    return '\n'.join(unindented_lines)



def main(args):

    dataset = datasets.load_dataset(args.dataset, data_dir=args.data_dir, revision=args.hf_commit_id, split="train")
    print(f"Loaded {len(dataset)} examples. Running pre-filtering...")


    # threads = os.cpu_count() - 1
    dataset = dataset.filter(pre_filtering, num_proc=1)

    model = LLM(args.model, dtype=auto_dtype(),
                gpu_memory_utilization=args.gpu_utilization, tensor_parallel_size=args.num_gpus)
    tokenizer = model.get_tokenizer()

    if args.sample_size is not None:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(args.sample_size))

    print(f"Now running stage 3 filtering on {len(dataset)} examples...")




    # Dummy prompt for token counting
    dummy = '''#' Dummy function
    dummy <- function() {
        return(NULL)
    }'''
    dummy_prompt = prompt_fmt(dummy)
    few_shot_toks = len(tokenizer.encode(
        dummy_prompt)) - len(tokenizer.encode(dummy))
    print(f"Few-shot prompt has {few_shot_toks} tokens")
    prompts = []
    for ex in tqdm(dataset, total=len(dataset), desc="Generating prompts"):
        code = ex[args.content_col]
        toks = len(tokenizer.encode(code)) + few_shot_toks
        if toks > 16380:
            print(f"Skipping example with {toks} tokens")
            prompts.append(dummy_prompt)
            continue
        p = prompt_fmt(code)
        prompts.append(p)

    responses = []
    for chunk in tqdm(chunkify(prompts, args.batch_size), desc="Generating responses"):
        outs = model.generate(chunk, SamplingParams(
            temperature=0.0, stop="\n", max_tokens=5))
        contents = [o.outputs[0].text for o in outs]
        for c in contents:
            yes_count = c.lower().count("yes")
            no_count = c.lower().count("no")
            if yes_count > no_count:
                responses.append(True)
            elif yes_count < no_count:
                responses.append(False)
            else:
                responses.append(False)

    new_ds = dataset.filter(
        lambda ex, i: responses[i] and "dummy <- function()" not in ex[args.content_col], with_indices=True)
    print(f"Filtered {len(dataset) - len(new_ds)} examples")
    # Rename 'content' column to 'seed'
    new_ds = new_ds.rename_column("content", "seed")
    new_ds.push_to_hub(args.push, private=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="TufanHossain/highquality_subset")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--hf_commit_id", type=str, default="69032b9117bd69ffb12227a6d6326c487892453e",
                        help="the version of the data from hugginface repository")
    parser.add_argument('--model', type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--gpu_utilization", type=float, default=0.6, help="GPU Memory Utilization")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--content_col', type=str, default="content")
    parser.add_argument('--push', type=str, default="TufanHossain/final_filtered_dataset")
    args = parser.parse_args()
    random.seed(42)
    main(args)