from datasets import load_dataset
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.rinterface
import logging
import os

# Suppress R environment warnings
os.environ["R_HOME"] = "/usr/lib/R"
os.environ["R_LIBS_USER"] = "/usr/local/lib/R/site-library"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def validate_r_code(response: str, tests: str, row_id: int) -> bool:
    try:
        # Initialize R environment
        rpy2.rinterface.initr()
        # Source the response code
        logger.info(f"Row {row_id}: Attempting to parse response:\n{response[:100]}...")
        ro.r(response)
        # Run the tests
        logger.info(f"Row {row_id}: Attempting to run tests:\n{tests[:100]}...")
        ro.r(tests)
        logger.info(f"Row {row_id}: Validation passed")
        return True
    except Exception as e:
        logger.error(f"Row {row_id}: Validation failed: {str(e)}")
        logger.error(f"Response snippet:\n{response[:200]}...")
        logger.error(f"Tests snippet:\n{tests[:200]}...")
        return False
    finally:
        # Clean up R environment
        ro.r('rm(list=ls())')

def main():
    # Load dataset
    dataset = load_dataset("TufanHossain/ossinstruct_r", "i_r_output", split="train")
    logger.info(f"Total rows: {len(dataset)}")

    # Validate each row
    valid_rows = []
    for i, row in enumerate(dataset):
        response = row['response']
        tests = row['tests']
        if validate_r_code(response, tests, i):
            valid_rows.append(row)

    # Save valid rows
    logger.info(f"Valid rows: {len(valid_rows)}/{len(dataset)}")
    if valid_rows:
        from datasets import Dataset
        valid_dataset = Dataset.from_list(valid_rows)
        valid_dataset.save_to_disk("valid_i_r_output")
    else:
        logger.warning("No valid rows to save")

if __name__ == "__main__":
    main()