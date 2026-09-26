"""Friendly messages for OpenAI API HTTP errors."""
import json


def friendly_openai_error(detail: str, status_code=None) -> str:
    """
    Turn raw OpenAI error JSON into a short clear message for the UI.
    """
    text = (detail or "").strip()
    lower = text.lower()
    code = ""
    api_msg = ""

    try:
        obj = json.loads(text)
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            api_msg = str(err.get("message") or "")
            code = str(err.get("code") or err.get("type") or "")
            lower = (api_msg + " " + code).lower()
    except Exception:
        api_msg = text

    if (
        "credit" in lower
        or "insufficient_quota" in lower
        or "credit_balance_exhausted" in lower
        or "billing" in lower
        or "quota" in lower
    ):
        return (
            "OpenAI credits are finished. "
            "Add credits at https://platform.openai.com/settings/organization/billing "
            "then restart the backend and try again."
        )

    if "invalid_api_key" in lower or "incorrect api key" in lower or "authentication" in lower:
        return (
            "OpenAI API key is invalid. "
            "Put a valid OPENAI_API_KEY in growsmartback/.env and restart the backend."
        )

    if "rate_limit" in lower or status_code == 429:
        return "OpenAI rate limit reached. Wait a minute and try again."

    if api_msg:
        return f"OpenAI error: {api_msg[:280]}"
    return f"OpenAI request failed: {text[:280]}"
