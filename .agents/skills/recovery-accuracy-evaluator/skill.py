import re
import math

def evaluate_recovery(original_error_type, reprompted_output, expected_output):
    expected_str = str(expected_output)
    actual_str = str(reprompted_output)

    # Try basic substring match
    if expected_str in actual_str:
        return {'recovered': True, 'notes': f'Successfully resolved {original_error_type}'}

    # Fallback to number extraction mapping
    expected_nums = re.findall(r"[\d,]+(?:\.\d+)?", expected_str)
    actual_nums = re.findall(r"[\d,]+(?:\.\d+)?", actual_str)

    # Float comparison with tolerance replaces fragile substring and string-based number matching.
    if expected_nums and all(
        any(math.isclose(float(e.replace(',','')), float(a.replace(',','')), abs_tol=0.1) for a in actual_nums)
        for e in expected_nums
    ):
        return {'recovered': True, 'notes': f'Successfully resolved {original_error_type}'}

    return {'recovered': False, 'notes': f'Failed to resolve {original_error_type}'}
