Three-Step Pipeline


Step 1: Seed Gathering

Purpose: Collect raw R-function examples and extract their content into a unified dataset.
Primary File: SeedGathering.ipynb


Step 2: Self-OSS-Instruct

Purpose: Generate in-context instruction–completion pairs from the seeds.
Primary File: self_ossinstruct.py (in root)



Step 3: Self-Validation

Purpose: Validate model-generated R code against embedded tests, retaining only passing examples.
Primary File: validate_i_r_output.py



File Summary

•	SeedGathering.ipynb – Notebook for raw data collection with three sub-steps and the critical download_contents function.

•	self_ossinstruct.py – Script to transform seeds into instruction–response pairs using vLLM.

•	validate_i_r_output.py – Script to run and validate generated R code via rpy2.

•	tree_sitter_parser.py – Compiles and loads the Tree‑sitter R grammar; helper functions: make_parser(), get_fn_name(), does_have_return().

•	autoimports_r.py – Infers missing library() calls in R code.

•	generate_from_the_stack_v2.py – Streams R files from Software Heritage to build a raw HF dataset.

•	filter_dataset.py – Applies heuristics and vLLM checks to refine docstring quality.

•	high_quality_subset.py – Further filters for explicit returns and style-checked functions.

•	star_align/ – Folder with alignment scripts for StarCoder2 outputs.
