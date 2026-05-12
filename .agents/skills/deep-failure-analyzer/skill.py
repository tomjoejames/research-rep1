import re

def analyze_failure(prompt, expected_output, actual_output):
    if not actual_output or not str(actual_output).strip():
        return "Context Collapse"
        
    expected_str = str(expected_output)
    actual_str = str(actual_output)
    
    expected_nums = re.findall(r"[\d,]+(?:\.\d+)?", expected_str)
    actual_nums = re.findall(r"[\d,]+(?:\.\d+)?", actual_str)
    
    # Check if all expected numbers are present in the actual output
    has_correct_numbers = True
    if expected_nums:
        for num in expected_nums:
            if num not in actual_nums:
                has_correct_numbers = False
                break
    else:
        # If there are no expected numbers, we can't reliably check just by numbers
        has_correct_numbers = False
        
    if has_correct_numbers:
        return "Formatting Error"
        
    if actual_nums and not has_correct_numbers:
        return "Calculation Error"
        
    return "Unknown Hallucination"
