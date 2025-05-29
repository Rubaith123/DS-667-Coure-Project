from tree_sitter_parser import LANGUAGE, make_parser, node_to_string
import datasets
import os
import signal
from multiprocessing import Pool
import argparse
import boto3
from smart_open import open
from botocore import UNSIGNED
from botocore.client import Config
from datasets import load_dataset


# Corrected function definition query for R
# TOPLEVEL_R_DOCSTRING_QUERY = LANGUAGE.query("""
# (
#   (binary_operator
#     lhs: (identifier) @function_name
#     operator: [ "<-" "=" ]
#     rhs: (function_definition
#       body: (braced_expression
#         (string) @docstring
#         (_)  ; Allow other expressions in the block
#       )
#     )
#   )
# )
# """)

# # Corrected function definition query for R
# TOPLEVEL_R_DOCSTRING_QUERY = LANGUAGE.query("""
# (
#   (binary_operator
#     lhs: (identifier) @function_name
#     operator: [ "<-" "=" "<<-" ]
#     rhs: (function_definition)
#   )
# )
# """)



# Define the R language and query
TOPLEVEL_R_DOCSTRING_QUERY = LANGUAGE.query("""
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




# # Corrected function definition query for R
# TOPLEVEL_R_DOCSTRING_QUERY = LANGUAGE.query("""
# (
#   (binary_operator
#     lhs: (identifier) @function_name
#     operator: [ "<-" "=" ]
#     rhs: (function_definition
#       body: (braced_expression
#         (string) @docstring
#         (_)  ; Allow other expressions in the block
#       )
#     )
#   )
# )
# """)



def download_contents(blob_id, src_encoding="utf-8"):
    # Set up the S3 client with unsigned access
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)

    return content




# def get_fns_with_docstrings(src, tree):
#     """
#     Extract function docstrings from R code.
#     """
#     captures = TOPLEVEL_R_DOCSTRING_QUERY.captures(tree.root_node)
#     res = []
#     for node, capture_name in captures:
#         if capture_name == "function_name":
#             # Check if it is a top-level function. In R, we check
#             # if the function definition is directly under the root
#             # or a top-level expression.
#             res.append(node_to_string(src, node))
#     return res


def get_fns_with_docstrings(src, tree):
    """
    Extract R functions with all preceding roxygen2 docstrings (consecutive #' lines).
    """
    captures = TOPLEVEL_R_DOCSTRING_QUERY.captures(tree.root_node)
    res = []

    # Convert source to lines (for easy slicing)
    if isinstance(src, bytes):
        src_str = src.decode("utf-8")
    else:
        src_str = src
    src_lines = src_str.splitlines()

    # Gather all function nodes (ignore comment captures)
    function_nodes = [node for node, cap in captures if cap == "function.def"]

    for fn_node in function_nodes:
        start_line, _ = fn_node.start_point  # zero-based line index

        # Collect consecutive roxygen lines before start_line
        docstring_lines = []
        line_idx = start_line - 1
        while line_idx >= 0:
            line = src_lines[line_idx].strip()
            if line.startswith("#'"):
                docstring_lines.insert(0, src_lines[line_idx])  # prepend to keep order
                line_idx -= 1
            else:
                break

        # Extract function code lines
        fn_start_line = start_line
        fn_end_line = fn_node.end_point[0]
        fn_lines = src_lines[fn_start_line:fn_end_line+1]

        # Combine docstring + function code
        combined_lines = docstring_lines + fn_lines
        combined_code = "\n".join(combined_lines)
        res.append(combined_code)

    return res







def parse_ex(parser, ex):
    """
    Parse a single R file from the dataset.
    """
    # ex = ex["content"]
    ex = download_contents(ex["blob_id"], ex["src_encoding"])
    try:
        buf = bytes(ex, "utf8")
        tree = parser.parse(buf)
        return get_fns_with_docstrings(buf, tree)
    except Exception as e:
        print(f"Error parsing example: {e}")
        return []


# Global parser instances
PARSERS = None


def process_chunk(idx_and_chunk):
    assert PARSERS is not None
    idx, chunk = idx_and_chunk
    parser = PARSERS[idx]
    chunk_new_funs = set()
    for ex in chunk:
        chunk_new_funs.update(parse_ex(parser, ex))
    return chunk_new_funs



def main(args):
    global PARSERS
    # Load dataset with dynamic sample size
    #ds = datasets.load_dataset(args.dataset, data_dir=args.data_dir, split="train")
    ds = datasets.load_dataset(args.dataset, data_dir=args.data_dir, split="train[:{}]".format(args.num_samples))
    #ds = ds.filter(lambda x: x["path"].endswith((".R", ".r", ".Rmd")))
    funs = set()
    PARSERS = [make_parser() for _ in range(args.num_workers)]
    total_len = len(ds)
    CHUNK_SIZE = args.chunk_size_factor * args.num_workers

    print(f"Total length: {total_len}")
    print(f"Chunk size: {CHUNK_SIZE}")

    chunk = []
    p = Pool(args.num_workers)
    for i, ex in enumerate(ds):
        if i % (total_len // 100) == 0:
            print(f"{i}/{total_len}")
        try:
            chunk.append(ex)
            if len(chunk) == CHUNK_SIZE or i == total_len - 1:
                print(f"Processing chunk {i // CHUNK_SIZE}")

                # Divide the chunk into NUM_WORKERS chunks
                subchunk_size = len(chunk) // args.num_workers
                subchunks = [chunk[i:i + subchunk_size]
                             for i in range(0, len(chunk), subchunk_size)]

                # Process the subchunks in parallel
                new_funs_iter = p.imap(
                    process_chunk, [(i, subchunk) for i, subchunk in enumerate(subchunks)]
                )

                print("Getting new functions")
                len_before = len(funs)

                while True:
                    try:
                        def timeout_handler(_, __):
                            raise KeyboardInterrupt  # Soft exit on timeout

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)
                        funs.update(next(new_funs_iter))
                        signal.alarm(0)
                    except KeyboardInterrupt:
                        signal.alarm(0)
                        print("Keyboard interrupt. Terminating pool")
                        p.terminate()
                        p = Pool(args.num_workers)
                        break
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error during function extraction: {e}")

                signal.alarm(0)

                # Recreate parsers for stability
                PARSERS = [make_parser() for _ in range(args.num_workers)]

                print(
                    f"Done processing chunk {i // CHUNK_SIZE}. Got {len(funs) - len_before} new functions"
                )

                chunk = []
        except Exception as e:
            print(f"Error processing chunk: {e}")
            chunk = []

        if i == total_len - 1:
            break

    p.close()

    # Create and push the new dataset to Hugging Face
    new_ds_dict = {
        "content": list(funs),
        "id": list(range(len(funs)))
    }

    new_ds = datasets.Dataset.from_dict(new_ds_dict)
    new_ds.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    # Modify the argument parser to accept --num_samples
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-v2")
    parser.add_argument("--data_dir", type=str, default="data/R")
    parser.add_argument("--num_samples", type=int, default=10000)  # Specify number of samples to load
    parser.add_argument("--chunk_size_factor", type=int, default=500, help="Factor to calculate chunk size")
    parser.add_argument("--push", type=str, default="TufanHossain/r-fns")

    args = parser.parse_args()
    main(args)
