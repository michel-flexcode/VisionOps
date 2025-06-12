VALID_CAR_BRANDS = {
    "Toyota",
    "BMW",
    "Audi",
    "Mercedes",
    "Tesla",
    "Volkswagen",
    "Honda",
    "Ford",
    "Chevrolet",
}

def parse_llm_response_and_verify(llm_response: str):
    """
    Parses the LLM text response to extract possible car brands mentioned,
    then verifies if they are in the known valid brands list.

    Args:
        llm_response (str): Raw text output from the LLM.

    Returns:
        list: Valid car brands found in the response.
    """
    found_brands = []
    # Simple splitting by commas, newlines or spaces - adjust if needed
    candidates = [token.strip() for token in llm_response.replace("\n", ",").split(",")]
    
    for candidate in candidates:
        for brand in VALID_CAR_BRANDS:
            if brand.lower() in candidate.lower():
                found_brands.append(brand)
                break

    return list(set(found_brands))  # Remove duplicates


def verification_items_from_llm_response(items_list):
    """
    Verify if each item in the list is a valid car brand.

    Args:
        items_list (list): List of strings (car brand candidates).

    Returns:
        list: Validated car brands only.
    """
    return [item for item in items_list if item in VALID_CAR_BRANDS]
