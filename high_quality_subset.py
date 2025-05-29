import datasets
import subprocess
import tempfile
import signal
import hashlib
import os
import argparse
from typing import List, Dict
from tqdm import tqdm
from huggingface_hub import HfApi

from tree_sitter_parser import LANGUAGE, global_parser

# Query for return statements in R (adapted for R)
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



def does_have_return(src):
    """
    Check if the given R code has a return statement.
    """
    tree = global_parser.parse(bytes(src, "utf8"))
    root = tree.root_node
    captures = RETURN_QUERY.captures(root)
    for node, _ in captures:
        # In R, a return statement might include an expression after "return"
        if len(node.children) <= 1:  # includes "return" itself
            continue
        else:
            return True

    return False



def run_r_linter(d):
    filemap = {}
    try:
        r_files = [f for f in os.listdir(d) if f.endswith(".R")]
        for file_name in r_files:
            try:
                process = subprocess.run(
                    ["Rscript", "-e", f"lintr::lint('{file_name}')"],  # Use just the filename
                    cwd=d,
                    capture_output=True,
                    timeout=120,
                    text=True,
                )
                outs = process.stdout
                errs = process.stderr
                if errs:
                    print(f"Linter errors (stderr):\n{errs}")

                lines = outs.splitlines()
                for i, line in enumerate(lines):
                    if ":" in line:
                        filepath = line.split(":")[0]
                        filename = filepath.split("/")[-1]
                        if filename not in filemap:
                            filemap[filename] = 0
                        #if "style:" in line or "warning:" in line or "error:" in line:
                        if "error:" in line:
                            filemap[filename] += 1

            except Exception as e:
                print(e)
                return None
    except Exception as e:
        print(e)
        return None

    return filemap



def typecheck_batch(files: List[str]) -> Dict[str, str]:
    # Create a temporary directory using the tempfile module
    filemap: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for contents in files:
            hash_object = hashlib.sha1(bytes(contents, "utf8"))
            hex_dig = hash_object.hexdigest()
            filemap[hex_dig] = contents
            name = os.path.join(tempdir, hex_dig + ".R")  # Change file extension to R for R code
            with open(name, "w") as f:
                f.write(contents)

        # Run pyright in the temporary directory
        typecheck_map = run_r_linter(tempdir)
        if typecheck_map is None:
            return {}

    for contents, errors in typecheck_map.items():
        no_r = contents.replace(".R", "")
        if errors == 0:
            continue

        if no_r in filemap:
            del filemap[no_r]

    print(f"Pass rate: {len(filemap)}/{len(files)}")

    return filemap


def infer_imports(code: str) -> str:
    from autoimports_r import autoimports_r

    try:
        def handler(signum, frame):
            raise Exception("Timeout")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        inferred = autoimports_r(code)
        signal.alarm(0)
        return inferred
    except Exception as e:
        signal.alarm(0)
        print(f"Error while inferring imports: {e}")
        return code


def main(args):
    ds = datasets.load_dataset(args.dataset,data_dir=args.data_dir, revision=args.hf_commit_id, split="train")

    print("Filtering to only functions with return statements")
    ds = ds.filter(lambda ex: does_have_return(
        ex["content"]), num_proc=os.cpu_count())

    if args.infer_imports:
        print("Inferring imports for functions")
        ds = ds.map(lambda ex: {"content": infer_imports(
            ex["content"])}, num_proc=os.cpu_count())
        # Remove examples where infer_imports returned None (e.g., due to unknown functions)
        ds = ds.filter(lambda ex: ex["content"] is not None, num_proc=os.cpu_count())

    batch = []
    max_i = len(ds) - 1

    new_ds = {
        "content": [],
        "sha1": [],
        "id": [],
    }

    e_id = 0

    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        try:
            code = ex["content"]

            batch.append(code)

            if len(batch) == args.batch_size or i == max_i:
                filemap = typecheck_batch(batch)
                for sha1, contents in filemap.items():
                    new_ds["content"].append(contents)
                    new_ds["sha1"].append(sha1)
                    new_ds["id"].append(e_id)
                    e_id += 1

                batch = []
        except Exception as e:
            print(f"There was an error: {e}")
            continue

    new_ds_hf = datasets.Dataset.from_dict(new_ds)
    new_ds_hf.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TufanHossain/r-fns", help="Points to dataset of R functions with return statements. Columns: 'content'")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--hf_commit_id", type=str, default="7000f8ba1e10aadd65c3b984a4eaee51870d1a05", help="the version of the data from hugginface repository")
    parser.add_argument("--push", type=str, default="TufanHossain/highquality_subset", help="Push to this dataset to which repo")
    parser.add_argument("--infer_imports", action="store_true", default=False, help="Infer imports for functions")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size for typechecking")
    args = parser.parse_args()
    main(args)
