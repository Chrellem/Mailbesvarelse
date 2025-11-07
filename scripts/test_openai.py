import os
import sys
import traceback
try:
    import openai
except Exception:
    openai = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def create_chat_completion_local(messages, model="gpt-3.5-turbo"):
    try:
        OpenAI = getattr(openai, "OpenAI", None)
        if OpenAI:
            client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1)
            return resp
    except Exception:
        pass
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=1)
        return resp
    except Exception as e:
        raise

if __name__ == "__main__":
    print("OPENAI_API_KEY set:", bool(OPENAI_API_KEY))
    if not OPENAI_API_KEY:
        print("Ingen nøgle - sæt OPENAI_API_KEY i env eller .streamlit/secrets.toml")
        sys.exit(1)
    try:
        messages = [{"role":"user","content":"Ping"}]
        resp = create_chat_completion_local(messages)
        print("Test OK. Response repr (type):", type(resp))
        try:
            text = getattr(resp, "choices", [])[0].message.content if hasattr(resp, "choices") and resp.choices else None
            if not text:
                text = resp["choices"][0]["message"]["content"]
            print("Sample text (truncated):", (text or "")[:200])
        except Exception:
            print("Kunne ikke læse tekstfelt fra respons. Resp keys/attrs:")
            try:
                print(dir(resp)[:50])
            except Exception:
                print(str(resp)[:500])
    except Exception as e:
        print("ERROR:", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)
