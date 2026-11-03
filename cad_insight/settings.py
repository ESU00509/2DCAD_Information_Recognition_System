import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
for folder in (DATA / 'jobs', DATA / 'cache', ROOT / '.cache'):
    folder.mkdir(parents=True, exist_ok=True)
# Model caches stay with the project, including in the dedicated environment.
os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(ROOT / '.cache' / 'paddlex'))
os.environ.setdefault('PADDLE_HOME', str(ROOT / '.cache' / 'paddle'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('LANGSMITH_TRACING', 'false')

@dataclass
class Settings:
    api_key: str = field(default='', repr=False)
    base_url: str = 'https://api.deepseek.com'
    model: str = 'deepseek-flash'
    timeout: float = 90
    max_requests: int = 8
    max_output_tokens: int = 10000
    thinking: bool = False
    input_price: float | None = None
    output_price: float | None = None

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env', override=True)
        def price(name):
            value = os.getenv(name, '').strip()
            return float(value) if value else None
        return cls(api_key=os.getenv('DEEPSEEK_API_KEY', ''),
                   base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                   model=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'),
                   timeout=float(os.getenv('DEEPSEEK_TIMEOUT', '90')),
                   max_requests=max(1, min(30, int(os.getenv('DEEPSEEK_MAX_REQUESTS', '8')))),
                   max_output_tokens=max(1024, min(32768, int(os.getenv('DEEPSEEK_MAX_OUTPUT_TOKENS', '10000')))),
                   thinking=os.getenv('DEEPSEEK_THINKING', 'disabled').lower() == 'enabled',
                   input_price=price('DEEPSEEK_INPUT_PRICE'), output_price=price('DEEPSEEK_OUTPUT_PRICE'))
