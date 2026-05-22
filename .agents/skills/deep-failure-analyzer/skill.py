import re
import math

def analyze_failure(prompt, expected_output, actual_output):
    if not actual_output or not str(actual_output).strip():
        return "Context Collapse"

    expected_str = str(expected_output)
    actual_str = str(actual_output)

    expected_nums = re.findall(r"[\d,]+(?:\.\d+)?", expected_str)
    actual_nums = re.findall(r"[\d,]+(?:\.\d+)?", actual_str)

    # Check if all expected numbers are present in the actual output
    # Use float comparison with tolerance instead of string match to handle rounding/formatting variants.
    has_correct_numbers = True
    if expected_nums:
        for num in expected_nums:
            try:
                if not any(math.isclose(float(num.replace(',','')), float(a.replace(',','')), abs_tol=0.1) for a in actual_nums):
                    has_correct_numbers = False
                    break
            except ValueError:
                has_correct_numbers = False
                break
    else:
        has_correct_numbers = False

    if has_correct_numbers:
        return "Formatting Error"

    if actual_nums and not has_correct_numbers:
        return "Calculation Error"

    return "Unknown Hallucination"
