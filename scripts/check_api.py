"""List account model capabilities without printing credentials or request headers."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad_insight.settings import Settings
from openai import OpenAI, APIStatusError, APIConnectionError

settings = Settings.load()
with OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=30, max_retries=0) as client:
    try:
        response = client.models.list()
        print(json.dumps({'models': [m.model_dump() for m in response.data]}, ensure_ascii=False))
    except APIStatusError as exc:
        print(json.dumps({'status': exc.status_code, 'type': type(exc).__name__}))
        sys.exit(1)
    except APIConnectionError:
        print('Connection failed; no credentials logged.')
        sys.exit(1)
