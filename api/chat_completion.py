from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class OpenAIAPI:
    def __init__(self, api_key,
                 model="gpt-4-0613",
                 api_base="http://localhost:9581/v1"
                 ):

        self.api_key = api_key
        self.model = model
        self.api_base = api_base

    def chat_completion(self, messages, temperature=0.7, **kwargs):
        return self.chat_completion_with_backoff(messages=messages,
                                                 temperature=temperature,
                                                 **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(50), reraise=False)
    def chat_completion_with_backoff(self, messages, temperature, **kwargs):
        import openai
        if "gpt" in self.model:
            if openai.__version__ > '1.0.0':
                client = openai.AzureOpenAI(
                    azure_endpoint=self.api_base,
                    api_version="2023-07-01-preview",
                    api_key=self.api_key
                )
                return client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs
                )
            else:
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                openai.api_type = "azure"
                openai.api_version = "2023-07-01-preview"
                return openai.ChatCompletion.create(
                    engine=self.model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs
                )

        else:
            openai.api_key = 'none'
            openai.api_base = self.api_base
            return openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
