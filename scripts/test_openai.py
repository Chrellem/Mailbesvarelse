import os
import sys

print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("openai imported, version:", getattr(openai, "__version__", "unknown"))
    # test call (brug en billig model til test)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1
    )
    print("OpenAI test OK, response keys:", list(resp.keys()))
except Exception as e:
    print("ERROR:", type(e).__name__, e)
    sys.exit(1)
