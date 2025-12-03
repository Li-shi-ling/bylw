from util.tool import convert_chinese_punctuation_to_english, find_text_intervals, align_extractions
from langextract.core import types as core_types
from typing import Any, Iterator, Sequence
from langextract.providers import patterns
from langextract.providers import router
from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import data
import concurrent.futures
import langextract as lx
import dataclasses
import http.client
import textwrap
import logging
import yaml
import json
import os

logging.basicConfig(level=logging.DEBUG)

patterns.DEEPSEEK_PATTERNS = [
    r'^deepseek-',
    r'deepseek',
]
patterns.DEEPSEEK_PRIORITY = 50

@router.register(
    *patterns.DEEPSEEK_PATTERNS,
    priority=patterns.DEEPSEEK_PRIORITY,
)
@dataclasses.dataclass(init=False)
class DeepSeekLanguageModel(base_model.BaseLanguageModel):
    """Language model inference using DeepSeek's API with structured output."""

    model_id: str = 'deepseek-chat'
    api_key: str | None = None
    base_url: str | None = None
    format_type: data.FormatType = data.FormatType.JSON
    temperature: float | None = None
    max_tokens: int = 2000
    max_workers: int = 10
    _client: Any = dataclasses.field(default=None, repr=False, compare=False)
    _extra_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def requires_fence_output(self) -> bool:
        """DeepSeek JSON mode returns raw JSON without fences."""
        if self.format_type == data.FormatType.JSON:
            return self.fence_output
        return super().requires_fence_output

    def __init__(
            self,
            model_id: str = 'deepseek-chat',
            api_key: str | None = None,
            base_url: str | None = None,
            format_type: data.FormatType = data.FormatType.JSON,
            temperature: float | None = None,
            max_tokens: int = 2000,
            max_workers: int = 10,
            fence_output: bool = True,
            **kwargs,
    ) -> None:
        """Initialize the DeepSeek language model.

        Args:
            model_id: The DeepSeek model ID to use (e.g., 'deepseek-chat', 'deepseek-coder').
            api_key: API key for DeepSeek service.
            base_url: Base URL for DeepSeek service (defaults to official API).
            format_type: Output format (JSON or YAML).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            max_workers: Maximum number of parallel API calls.
            **kwargs: Ignored extra parameters so callers can pass a superset of
                arguments shared across back-ends without raising ``TypeError``.
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url or "api.deepseek.com"
        self.format_type = format_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.fence_output = fence_output

        if not self.api_key:
            raise exceptions.InferenceConfigError('DeepSeek API key not provided.')

        # Initialize the DeepSeek client
        self._client = DeepSeekAPIClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model_id=self.model_id
        )

        super().__init__(
            constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
        )
        self._extra_kwargs = kwargs or {}

    def _process_single_prompt(
            self, prompt: str, config: dict
    ) -> core_types.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            system_message = ''
            if self.format_type == data.FormatType.JSON:
                system_message = (
                    'You are a helpful assistant that responds in JSON format.'
                )
            elif self.format_type == data.FormatType.YAML:
                system_message = (
                    'You are a helpful assistant that responds in YAML format.'
                )

            # Build messages
            messages = [{"role": "user", "content": prompt}]
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})

            # Prepare API parameters
            api_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": config.get('max_tokens', self.max_tokens),
                "temperature": config.get('temperature', self.temperature or 0.0),
                "stream": False
            }

            # Add optional parameters
            for key in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
                if key in config:
                    api_params[key] = config[key]

            # DeepSeek API call
            response = self._client.chat_completions_create(**api_params)

            # Extract response text
            output_text = response["choices"][0]["message"]["content"]

            return core_types.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'DeepSeek API error: {str(e)}', original=e
            ) from e

    def infer(
            self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Runs inference on a list of prompts via DeepSeek's API.

        Args:
            batch_prompts: A list of string prompts.
            **kwargs: Additional generation params (temperature, top_p, etc.)

        Yields:
            Lists of ScoredOutputs.
        """
        merged_kwargs = self.merge_kwargs(kwargs)

        config = {}

        temp = merged_kwargs.get('temperature', self.temperature)
        if temp is not None:
            config['temperature'] = temp

        config['max_tokens'] = merged_kwargs.get('max_tokens', self.max_tokens)

        for key in [
            'top_p',
            'frequency_penalty',
            'presence_penalty',
            'stop',
            'max_output_tokens'
        ]:
            if key in merged_kwargs:
                config[key] = merged_kwargs[key]

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[core_types.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise exceptions.InferenceRuntimeError(
                            f'Parallel inference error: {str(e)}', original=e
                        ) from e

                for result in results:
                    if result is None:
                        raise exceptions.InferenceRuntimeError(
                            'Failed to process one or more prompts'
                        )
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]

class DeepSeekAPIClient:
    """HTTP client for DeepSeek API."""

    def __init__(self, api_key: str, base_url: str = "api.deepseek.com", model_id: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions_create(self, **kwargs) -> dict:
        """Create a chat completion using DeepSeek API."""
        conn = http.client.HTTPSConnection(self.base_url)

        # Prepare payload
        payload = {
            "model": kwargs.get("model", self.model_id),
            "messages": kwargs.get("messages", []),
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "stream": False
        }

        # Add optional parameters
        optional_params = ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        # Make request
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), self.headers)
        response = conn.getresponse()

        if response.status != 200:
            raise exceptions.InferenceRuntimeError(
                f"DeepSeek API returned status {response.status}: {response.reason}"
            )

        response_data = response.read().decode("utf-8")
        conn.close()
        # print(response_data)
        return json.loads(response_data)

class LangExtract:
    def __init__(
            self,
            DEEPSEEK_API_KEY,
            prompt = None,
            examples = None,

    ):
        self.deepseek_model = DeepSeekLanguageModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            temperature=0.0,
            max_tokens=2000
        )
        self.prompt = prompt
        self.examples = examples

    def Setprompt(self, prompt):
        self.prompt = prompt

    def Setexamples(self, examples):
        self.examples = examples

    def extract(self, input_text, outputputh):
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=self.prompt,
            examples=self.examples,
            model=self.deepseek_model,
            fence_output=True
        )
        lx.io.save_annotated_documents(
            [result],  # 将结果包装成列表传入
            output_name=outputputh  # 输出文件名
        )
        with open("./test_output/extraction_results.jsonl", "r") as f:
            data = json.loads(f.read())
        result = align_extractions(data)
        with open('./test_output/extraction_results.jsonl', 'w', encoding='utf-8') as f:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

    def jsonltoweb(self, inputputh, outputputh):
        html_content = lx.visualize(inputputh)
        with open(outputputh, "w", encoding="utf-8") as f:
            if hasattr(html_content, "data"):
                f.write(html_content.data)
            else:
                f.write(html_content)

if __name__ == "__main__":
    import time
    s_t = time.time()

    with open('../../config/base.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    os.environ['DEEPSEEK_API_KEY'] = config["DeepSeek"]["DEEPSEEK_API_KEY"]

    prompt = textwrap.dedent("""\
    请严格按照以下要求从文本中提取信息：

    提取类别：
    1. character (角色) - 故事中的人物
    2. emotion (情感) - 表达的情感或情绪  
    3. relationship (关系) - 角色之间的关系

    提取规则：
    - 使用文本中的确切原文，不要转述
    - 按出现顺序提取
    - 为每个提取项提供相关属性
    - 不要重叠实体

    输出格式请使用清晰的标记格式。
    """)

    examples = [
        lx.data.ExampleData(
            text=(
                "ROMEO. But soft! What light through yonder window breaks? It is "
                "the east, and Juliet is the sun. Juliet appears at the window."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"},
                ),
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="Juliet",
                    attributes={"description": "the sun"},
                ),
                lx.data.Extraction(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"},
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"},
                ),
            ],
        )
    ]

    fence_output = True

    deepseek_model = DeepSeekLanguageModel(
        model_id="deepseek-chat",
        api_key=config["DeepSeek"]["DEEPSEEK_API_KEY"],
        temperature=0.1,
        max_tokens=3000,
        top_p=0.9,
        fence_output = fence_output
    )
    with open("../../cs/KG测试.txt") as f:
        documents = lx.data.Document(text=f.read())

    # 修改 extract 调用，移除不支持的参数
    result = lx.extract(
        text_or_documents=[documents],
        prompt_description=prompt,
        examples=examples,
        model=deepseek_model,
        fence_output=fence_output,
    )
    lx.io.save_annotated_documents(
        result,  # 将结果包装成列表传入
        output_name="extraction_results.jsonl"  # 输出文件名
    )

    e_t = time.time()
    print(f"使用时间:{e_t - s_t}")