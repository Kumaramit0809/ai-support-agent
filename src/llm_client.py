"""Thin shared wrapper around the Groq API."""
from __future__ import annotations
import json, os, threading, time
from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
MIN_SECONDS_BETWEEN_CALLS = 2.2
MAX_RETRIES = 5
_rate_lock = threading.Lock()
_last_call_time = 0.0

def _throttle():
    global _last_call_time
    with _rate_lock:
        elapsed = time.monotonic() - _last_call_time
        wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()

def get_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com/keys")
    return Groq(api_key=api_key)

def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

def _call_with_retry(fn):
    import groq as groq_module
    last_err = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            return fn()
        except groq_module.RateLimitError as e:
            last_err = e
            wait = 20 * (attempt + 1)
            print(f"[llm_client] rate limited, waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except groq_module.APIStatusError as e:
            last_err = e
            if e.status_code and e.status_code < 500:
                raise
            wait = 10 * (attempt + 1)
            print(f"[llm_client] server error, waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise last_err

def generate_text(prompt: str, model: str = DEFAULT_MODEL, client=None) -> str:
    client = client or get_client()
    def _do():
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip()
    return _call_with_retry(_do)

def generate_json(prompt: str, model: str = DEFAULT_MODEL, client=None) -> dict | list:
    client = client or get_client()
    def _do():
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = _strip_json_fence(resp.choices[0].message.content or "")
        return json.loads(text)
    try:
        return _call_with_retry(_do)
    except Exception:
        def _do_no_mode():
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            text = _strip_json_fence(resp.choices[0].message.content or "")
            return json.loads(text)
        return _call_with_retry(_do_no_mode)