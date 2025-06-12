import requests

def send_description_to_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:1234/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 450,
                "temperature": 0.0
            },
            timeout=600
        )
        response.raise_for_status()

        data = response.json()

        # Check that the response contains what we expect
        if "choices" in data and len(data["choices"]) > 0 and "text" in data["choices"][0]:
            return data["choices"][0]["text"].strip()
        else:
            return "Error: LLM response format is invalid or incomplete."

    except requests.exceptions.RequestException as e:
        return f"Error communicating with the LLM: {e}"
