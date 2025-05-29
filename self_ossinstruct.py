import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from openai.types import CompletionChoice, Completion
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser
from huggingface_hub import HfApi, login

import star_align
from tree_sitter_parser import get_fn_name, does_have_return

InstructMode = Literal["I->R", "S->C", "C->I", "S->I"]

LANGUAGE_MAP = {
    "cpp": "C++",
    "java": "Java",
    "php": "PHP",
    "python": "Python",
    "rust": "Rust",
    "typescript": "TypeScript",
    "r": "R"
}


LLAMA3 = os.getenv("LLAMA3") is not None

if LLAMA3:
    print("Using Llama-3 prompt format")

def flatten_openai_responses(responses: list[Completion]) -> list[Completion]:
    completions = []
    for idx, response in enumerate(responses):
        completions.extend(
            Completion(
                id=f"{response.id}:{idx}",
                created=response.created,
                object=response.object,
                model=response.model,
                choices=[choice],
                system_fingerprint=response.system_fingerprint,
            )
            for choice in response.choices
        )
    return completions

@dataclass(frozen=True)
class Args:
    max_new_data: int
    instruct_mode: InstructMode
    seed_dataset: str = field(default="TufanHossain/final_filtered_dataset", metadata={"help": "Hugging Face dataset name"})
    model: str = field(default="bigcode/starcoder2-3b")  # Updated to 3B model
    data_dir: str = field(default="data", metadata={"help": "Data directory for dataset"})
    hf_commit_id: str = field(default="main", metadata={"help": "Hugging Face dataset commit ID"})
    hf_repo: str = field(default="TufanHossain/ossinstruct_r", metadata={"help": "Hugging Face repository to save dataset"})

    use_vllm_server: bool = field(default=True)
    seed_code_start_index: int = field(default=0)
    continue_from: str | None = field(default=None)

    seed: int = field(default=3407)
    temperature: float = field(default=0.7)
    max_output_tokens: int = field(default=1536)
    prompting_mode: Literal["chat", "completion"] = field(default="completion")
    num_fewshots: int = field(default=5)

    async_micro_batch_size: int = field(default=1)
    num_batched_requests: int = field(default=1)
    num_sample_per_request: int = field(default=32)
    sleep: float | None = field(default=None)
    delay: float | None = field(default=None)

    tag: str = field(default="")
    save_dir: str = field(default="./")
    save_local: bool = field(default=False)

    def fingerprint(self, fewshot: "Fewshot | None") -> str:
        args = (
            [self.seed_dataset],
            self.seed,
            self.prompting_mode,
            self.num_fewshots,
            self.temperature,
            self.model,
            self.max_output_tokens,
            fewshot,
        )
        return star_align.utils.compute_fingerprint(*args, hash_length=5)

@dataclass(frozen=True)
class Property:
    category: str
    language: str
    concepts: list[str]
    difficulty: str

    @staticmethod
    def random_exercise(concepts: list[str], language: str) -> "Property":
        category = random.choice(["function implementation", "data processing", "statistical analysis"])
        difficulty = random.choice(["easy", "medium", "hard"])
        return Property(category=category, language=language, concepts=concepts, difficulty=difficulty)

    def concepts_prompt(self) -> str:
        return ", ".join(self.concepts)

    def prompt(self) -> str:
        category = f"category: {self.category}"
        language = f"language: {self.language}"
        difficulty = f"difficulty: {self.difficulty}"
        concepts = f"concepts: {self.concepts_prompt()}"
        return "\n".join([category, language, difficulty, concepts])

    def to_json(self) -> dict[str, str | list[str]]:
        return dict(category=self.category, language=self.language, concepts=self.concepts, difficulty=self.difficulty)

    @staticmethod
    def from_json(data: dict) -> "Property":
        assert all(isinstance(data[key], str) for key in ["category", "language", "difficulty"])
        assert isinstance(data["concepts"], list)
        return Property(category=data["category"], language=data["language"], concepts=data["concepts"], difficulty=data["difficulty"])

@dataclass(frozen=True)
class Example:
    property: Property
    snippet: str
    instruction: str
    response: str
    tests: str

    @staticmethod
    def prefix_template(mode: InstructMode) -> str:
        if mode == "I->R":
            if LLAMA3:
                return "### Instruction {index}\n{instruction}\n\n### Response {index}\n"
            else:
                return "<instruction>\n{instruction}\n</instruction>\n\n<response>\n"
        elif mode == "S->C":
            return "### Snippet\n```R\n{snippet}\n```\n\n### Concepts\n"
        elif mode == "C->I":
            return "### Properties\n{property}\n\n### Task\n"
        elif mode == "S->I":
            return "### Snippet\n```R\n{snippet}\n```\n\n### Task\n"
        else:
            assert False

    def prompt(self, mode: InstructMode, return_in_separate: bool = False, index: int | None = None) -> str | tuple[str, str]:
        assert index is None or (mode == "I->R" and LLAMA3)
        if mode == "I->R":
            kwargs = dict(instruction=self.instruction)
            if LLAMA3:
                assert index is not None
                kwargs["index"] = str(index)
                suffix = f"{self.response}\n\n### Tests {index}\n{self.tests}"
            else:
                suffix = f"{self.response}\n</response>\n\n<tests>\n{self.tests}\n</tests>"
        elif mode == "S->C":
            kwargs = dict(snippet=self.snippet)
            suffix = self.property.concepts_prompt()
        elif mode == "C->I":
            property_prompt = self.property.prompt()
            kwargs = dict(property=property_prompt)
            suffix = self.instruction
        elif mode == "S->I":
            kwargs = dict(snippet=self.snippet)
            suffix = self.instruction
        else:
            assert False
        prefix = self.prefix_template(mode).format(**kwargs)
        if return_in_separate:
            return prefix, suffix
        else:
            return prefix + suffix

@dataclass(frozen=True)
class Fewshot:
    sys_i_r: str
    sys_c_i: str
    sys_s_c: str
    sys_s_i: str
    examples: list[Example]

    def system_prompt(self, mode: InstructMode) -> str:
        attr_name = "sys_" + mode.replace("->", "_").replace("-", "_").lower()
        return getattr(self, attr_name)

    def valid_examples(self, mode: InstructMode) -> list[Example]:
        return self.examples

    def random_prompt(self, mode: InstructMode, num_fewshots: int, prompting_mode: Literal["chat", "completion"], **format_args: str) -> str:
        valid_examples = self.valid_examples(mode)
        assert 0 < num_fewshots <= len(valid_examples), f"{num_fewshots=}, {len(valid_examples)=}"
        examples = random.sample(valid_examples, k=num_fewshots)
        body = "\n\n".join(
            f"## Example {idx + 1}\n{example.prompt(mode, index=idx + 1 if LLAMA3 and mode == 'I->R' else None)}"
            for idx, example in enumerate(examples)
        )
        prefix_template = Example.prefix_template(mode)
        if mode == "I->R" and LLAMA3:
            format_args["index"] = str(len(examples) + 1)
        prefix = f"## Example {len(examples) + 1}\n" + prefix_template.format(**format_args)
        system_prompt = self.system_prompt(mode)
        full_prompt = f"{system_prompt}\n\n{body}\n\n{prefix}"
        return full_prompt

def parse_property(content: str) -> Property | None:
    content = content.strip()
    lines = content.split("\n")
    if len(lines) != 4:
        return None
    try:
        lines = [line[line.index(":") + 1:].strip() for line in lines]
    except ValueError:
        return None
    category, language, difficulty, concepts_str = lines
    concepts = list(map(str.strip, concepts_str.split(",")))
    return Property(category, language, concepts, difficulty)

def get_ossinstruct_fewshots() -> Fewshot:
    content = Path("prompts/self-ossinstruct-fewshot.txt").read_text().strip()
    splits = re.split(r"### Example \d+", content)
    system_prompt = splits[0].strip()
    sys_pattern = r"### System: I->R|### System: C->I|### System: S->C|### System: S->I"
    _, i_r, c_i, s_c, s_i = list(map(str.strip, re.split(sys_pattern, system_prompt)))
    if LLAMA3:
        i_r = f"{i_r}\n\nFor each '## Example' below, provide a '### Response' and a '### Tests' section."
    examples_str = [example.strip() for example in splits[1:]]
    examples = []
    for example_str in examples_str:
        pattern = r"\[Code\]\n|\[Property\]\n|\[Instruction\]\n|\[Response\]\n|\[Tests\]\n"
        _, snippet, property, instruction, response, tests = re.split(pattern, example_str)
        snippet = snippet.rstrip()
        property = parse_property(property)
        assert property is not None
        instruction = instruction.strip()
        response = response.strip()
        tests = tests.strip()
        example = Example(property, snippet, instruction, response, tests)
        examples.append(example)
    return Fewshot(sys_i_r=i_r, sys_c_i=c_i, sys_s_c=s_c, sys_s_i=s_i, examples=examples)



# def parse_generated_content(content: str, instruct_mode: InstructMode) -> dict | None:
#     if instruct_mode == "I->R":
#         parts = content.split("### Tests")
#         if len(parts) != 2:
#             print(f"Invalid format, expected '### Tests': {content}")
#             return None
#         response, tests = parts
#         response = response.strip()
#         tests = tests.strip()
#         if not response or not tests:
#             print(f"Empty response or tests: {content}")
#             return None
#         # if not does_have_return(response):
#         #     return None
#         return dict(response=response, tests=tests)
#     elif instruct_mode == "S->C":
#         concepts = list(map(str.strip, content.split(",")))
#         return dict(concepts=concepts)
#     elif instruct_mode == "C->I":
#         return dict(instruction=content.strip())
#     elif instruct_mode == "S->I":
#         return dict(instruction=content.strip())
#     else:
#         assert False



# def parse_generated_content(content: str, instruct_mode: InstructMode) -> dict | None:
#     if instruct_mode == "I->R":
#         parts = content.split("### Tests")
#         response = parts[0].strip()
#         tests = parts[1].strip() if len(parts) > 1 else "No tests provided"
#         if not response:
#             print(f"Empty response: {content}")
#             return None
#         return dict(response=response, tests=tests)
#     elif instruct_mode == "S->C":
#         concepts = list(map(str.strip, content.split(",")))
#         return dict(concepts=concepts)
#     elif instruct_mode == "C->I":
#         return dict(instruction=content.strip())
#     elif instruct_mode == "S->I":
#         return dict(instruction=content.strip())
#     else:
#         assert False


# def parse_generated_content(content: str, instruct_mode: InstructMode) -> None | dict:
#     if instruct_mode == "I->R":
#         import re
#         # Extract response between <response> and </response>
#         response_match = re.search(r'<response>\n(.*?)\n</response>', content, re.DOTALL)
#         response = response_match.group(1).strip() if response_match else ""
#         # Extract tests between <tests> and </tests>
#         tests_match = re.search(r'<tests>\n(.*?)\n</tests>', content, re.DOTALL)
#         tests = tests_match.group(1).strip() if tests_match else "No tests provided"
#         if not response:
#             print(f"Empty response: {content}")
#             return None
#         # Validate R code (check for R-specific syntax)
#         if not re.search(r'\b(function\s*\(|<-|\[\[)', response):
#             print(f"Non-R code detected: {response[:100]}...")
#             return None
#         return dict(response=response, tests=tests)
#     elif instruct_mode == "S->C":
#         concepts = list(map(str.strip, content.split(",")))
#         return dict(concepts=concepts)
#     elif instruct_mode == "C->I":
#         return dict(instruction=content.strip())
#     elif instruct_mode == "S->I":
#         return dict(instruction=content.strip())
#     else:
#         assert False



def parse_generated_content(content: str, instruct_mode: InstructMode) -> None | dict:
    if instruct_mode == "I->R":
        import re
        # Extract response between <response> and </response>
        response_match = re.search(r'<response>\n(.*?)\n</response>', content, re.DOTALL)
        response = response_match.group(1).strip() if response_match else ""
        # Extract tests between <tests> and </tests>
        tests_match = re.search(r'<tests>\n(.*?)\n</tests>', content, re.DOTALL)
        tests = tests_match.group(1).strip() if tests_match else ""
        # Validate response
        if not response:
            print(f"Empty response: {content[:100]}...")
            return None
        # Check for roxygen2 docstring, function definition, and body
        if not (re.search(r"#\'\s*@param", response) and
                re.search(r'\bfunction\s*\(', response) and
                re.search(r'\{[\s\S]*\}', response)):
            print(f"Invalid R code (missing docstring, function, or body): {response[:100]}...")
            return None
        # Validate tests
        if not tests or not re.search(r'\b(stopifnot|tryCatch)\b', tests):
            print(f"Invalid or missing tests (need stopifnot/tryCatch): {tests[:100] or 'No tests'}...")
            return None
        return dict(response=response, tests=tests)
    elif instruct_mode == "S->C":
        concepts = list(map(str.strip, content.split(",")))
        return dict(concepts=concepts)
    elif instruct_mode == "C->I":
        return dict(instruction=content.strip())
    elif instruct_mode == "S->I":
        return dict(instruction=content.strip())
    else:
        assert False



def build_kwargs(instruct_mode: InstructMode, example: dict) -> dict[str, str]:
    kwargs = {}
    if instruct_mode == "I->R":
        instruction = example.get("instruction")
        if instruction is None and "parsing_result" in example:
            instruction = example["parsing_result"][0]["instruction"]
        if instruction is None:
            raise ValueError("No instruction found in example")
        kwargs["instruction"] = instruction
    elif instruct_mode in ["S->C", "S->I"]:
        kwargs["snippet"] = example["seed"]
    elif instruct_mode == "C->I":
        lang = example.get("data_dir", "r").lower()
        language = LANGUAGE_MAP.get(lang, "R")
        concepts = example.get("concepts")
        if concepts is None and "parsing_result" in example:
            concepts = example["parsing_result"][0]["concepts"]
        if concepts is None:
            raise ValueError("No instruction found in example")
        property = Property.random_exercise(concepts, language=language)
        property_prompt = property.prompt()
        kwargs["property"] = property_prompt
        kwargs["property_obj"] = property
    else:
        assert False
    return kwargs




# def get_readable_prefix(instruct_mode: InstructMode, example: dict) -> str:
#     mode_pattern = instruct_mode.split("->")[0]
#     mode_map = {"I": "Instruction", "S": "Seed", "R": "Response", "C": "Concepts", "P": "Property"}
#     prefix_parts = []
#     for mode in mode_pattern:
#         key = mode_map[mode].lower()
#         if mode == "C" and key not in example and "parsing_result" in example:
#             # For C->I, use concepts from parsing_result if concepts is missing
#             value = example["parsing_result"][0]["concepts"]
#         else:
#             value = example[key]
#         prefix_parts.append(f"@@@{mode_map[mode]}\n{value}")
#     prefix = "\n\n".join(prefix_parts)
#     return prefix


def get_readable_prefix(instruct_mode: InstructMode, example: dict) -> str:
    mode = instruct_mode.split("->")[0]  # "I" for I->R, "C" for C->I, "S" for S->C
    key = "concepts" if mode == "C" else "seed" if mode == "S" else "instruction"
    if mode in ["C", "I"] and key not in example and "parsing_result" in example:
        if example["parsing_result"] and isinstance(example["parsing_result"], list) and len(example["parsing_result"]) > 0:
            value = example["parsing_result"][0][key]
        else:
            print(f"Warning: Empty or invalid parsing_result in example: {example}")
            value = "Unknown"
    else:
        value = example.get(key, "Unknown")
    return f"@@@ {key.title()}\n{value}\n"



async def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    assert args.num_batched_requests % args.async_micro_batch_size == 0
    if args.async_micro_batch_size > 1:
        assert args.num_sample_per_request == 1, "Only support 1 sample with batched async requests"
    if args.use_vllm_server:
        openai_client = star_align.utils.OpenAIClient()

    # Load seed dataset
    raw_dataset = load_dataset(args.seed_dataset, data_dir=args.data_dir, revision=args.hf_commit_id, split="train")
    id_key = "seed"
    if os.getenv("IGNORE_SEED_CHECK") is None:
        assert len(set(d[id_key] for d in raw_dataset)) == len(raw_dataset), "Duplicate seeds"
    else:
        print("[Warning] Ignoring seed check")

    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(start_index, end_index))
    dataset = raw_dataset.to_list()

    fewshot = get_ossinstruct_fewshots()
    data_fingerprint = args.fingerprint(fewshot)
    timestamp = star_align.utils.timestamp()

    # Prepare output path
    if args.continue_from is not None:
        assert data_fingerprint in args.continue_from, f"Fingerprint mismatch: {data_fingerprint}"
        assert f"-{start_index}-" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = star_align.utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_seed = old_data[-1][id_key]
        seed_index = next(idx for idx, d in enumerate(dataset) if d[id_key] == last_seed)
        n_skipped = seed_index + 1
        print(f"Continuing from {old_path} with {n_skipped} seed snippets skipped")
        if args.save_local:
            f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        mode_str = args.instruct_mode.replace("->", "_").lower()
        path = Path(args.save_dir, f"data{tag}-{mode_str}-{data_fingerprint}-{start_index}-{timestamp}.jsonl")
        assert not path.exists()
        if args.save_local:
            f_out = path.open("w")
            print("Saving to", path)
        n_skipped = 0

    # Initialize data for Hugging Face
    output_data = []
    dataset = dataset[n_skipped:]
    chunked_dataset = list(star_align.utils.chunked(dataset, n=args.num_batched_requests))
    pbar = tqdm(chunked_dataset)
    n_succeeded = 0

    if not args.use_vllm_server:
        from vllm import LLM, SamplingParams, RequestOutput
        import torch
        engine = LLM(args.model, tensor_parallel_size=torch.cuda.device_count())
        def vllm_response_to_openai(response: RequestOutput) -> Completion:
            created = 0
            choices = []
            for output in response.outputs:
                choice = CompletionChoice(
                    text=output.text,
                    index=0,
                    finish_reason="stop" if output.finish_reason == "stop" else "length",
                )
                choices.append(choice)
            return Completion(id=response.request_id, created=created, object="text_completion", model="not-specified", choices=choices, system_fingerprint="None")

    for chunk_index, examples in enumerate(pbar):
        effective_index = chunk_index * args.num_batched_requests + start_index + n_skipped
        print("Effective index:", effective_index)
        if chunk_index > 0 and args.sleep is not None:
            print(f"Sleeping for {args.sleep} seconds...")
            time.sleep(args.sleep)
        request_params = []
        all_prompts = []
        for index, example in enumerate(examples):
            seed = args.seed + effective_index + index
            random.seed(seed)
            kwargs = build_kwargs(args.instruct_mode, example)
            prompt = fewshot.random_prompt(
                args.instruct_mode,
                args.num_fewshots,
                prompting_mode=args.prompting_mode,
                **kwargs,
            )
            prompt = prompt.rstrip()
            all_prompts.append(prompt)
            max_new_tokens = args.max_output_tokens
            params = dict(
                model=args.model,
                max_tokens=max_new_tokens,
                n=args.num_sample_per_request,
                temperature=args.temperature,
                seed=seed,
                prompt=prompt,
                stop=["## Example"]
            )
            if args.instruct_mode == "I->R":
                params["stop"].append("### Tests")
            request_params.append(params)
        assert len(request_params) == len(examples)
        print(f"Ready to make {len(request_params)} requests")
        if args.use_vllm_server:
            dispatch_requests = openai_client.dispatch_completions
            if args.async_micro_batch_size == 1:
                responses = await dispatch_requests(request_params, delay=args.delay)
            else:
                request_params_batched = []
                request_params_chunks = star_align.utils.chunked(request_params, args.async_micro_batch_size)
                for request_params_chunk in request_params_chunks:
                    request_param = {k: v for k, v in request_params_chunk[0].items() if k != "prompt"}
                    request_param["prompt"] = [req["prompt"] for req in request_params_chunk]
                    request_params_batched.append(request_param)
                n_async_chunks = args.num_batched_requests // args.async_micro_batch_size
                assert len(request_params_batched) in [n_async_chunks, n_async_chunks + 1]
                print(f"Ready to make {len(request_params_batched)} batched async requests")
                responses_batched = await dispatch_requests(request_params_batched, delay=args.delay)
                responses = flatten_openai_responses(responses_batched)
                assert len(responses) == len(examples)
        else:
            stop = ["## Example"]
            if args.instruct_mode == "I->R":
                stop.append("### Tests")
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
                seed=args.seed + effective_index,
                n=args.num_sample_per_request,
                stop=stop,
            )
            vllm_responses = engine.generate(all_prompts, sampling_params)
            responses = list(map(vllm_response_to_openai, vllm_responses))

        assert len(examples) == len(responses)
        for prompt, example, response in zip(all_prompts, examples, responses):
            if isinstance(response, BaseException):
                print("Exception when generating response:", response)
                continue
            fingerprint = response.system_fingerprint
            original_mapping = {k: v for k, v in example.items() if k not in ["prompt", "fingerprint"]}
            # success_parsing_res = []
            # for choice in response.choices:
            #     if choice.finish_reason in ["stop", "eos_token"]:
            #         content = choice.text
            #         parsing_result = parse_generated_content(content, args.instruct_mode)
            #         if parsing_result is None:
            #             print("[WRONG FORMAT]")
            #             print("@@@Prompt", prompt, sep="\n", end="\n\n")
            #             print("@@@Response", content, sep="\n", end="\n\n")
            #             continue
            #         success_parsing_res.append(parsing_result)
            #     else:
            #         print("Failed reason:", choice.finish_reason)

            success_parsing_res = []
            failed_choices = []
            for choice in response.choices:
                content = choice.text
                parsing_result = parse_generated_content(choice.text, args.instruct_mode)
                if parsing_result:
                    success_parsing_res.append(parsing_result)
                else:
                    failed_choices.append(choice.text)
                    print(
                        f"Failed output for instruction '{example['parsing_result'][0]['instruction'] if 'parsing_result' in example else example.get('instruction')}': {choice.text}")

            n_failed_samples = args.num_sample_per_request - len(success_parsing_res)
            print(f"✅ Success samples: {len(success_parsing_res)}")
            print(f"❌ Failed samples: {n_failed_samples}")
            if len(success_parsing_res) == 0:
                print("No successful choices")
                continue
            data = dict(prompt=prompt, fingerprint=fingerprint, **original_mapping)
            if args.num_sample_per_request > 1:
                data["parsing_result"] = success_parsing_res
                prefix = prompt
            else:
                assert len(success_parsing_res) == 1
                parsing_result = success_parsing_res[0]
                data = dict(**data, **parsing_result)
                prefix = get_readable_prefix(args.instruct_mode, example)
            print(
                "@@@Prefix", prefix, f"@@@Generation (1 example)", content, sep="\n", end="\n\n"
            )
            n_succeeded += 1
            if args.save_local:
                f_out.write(json.dumps(data) + "\n")
                f_out.flush()
            output_data.append(data)
        total_requests = chunk_index * args.num_batched_requests + len(examples)
        pbar.set_description(f"Success ratio: {n_succeeded} / {total_requests}")

    # Save to Hugging Face
    if output_data:
        mode_str = args.instruct_mode.replace("->", "_").lower()
        dataset_name = f"{mode_str}_output"
        print(f"Saving dataset to Hugging Face: {args.hf_repo}/{dataset_name}")
        try:
            hf_dataset = Dataset.from_list(output_data)
            hf_dataset.push_to_hub(args.hf_repo, dataset_name, private=False, token=os.getenv("HUGGINGFACE_TOKEN"))
            print(f"Successfully pushed dataset to {args.hf_repo}/{dataset_name}")
        except Exception as e:
            print(f"Failed to push dataset to Hugging Face: {e}")

if __name__ == "__main__":
    # Login to Hugging Face
    if os.getenv("HUGGINGFACE_TOKEN"):
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    asyncio.run(main())