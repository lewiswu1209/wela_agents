
import trafilatura

from typing import Any
from typing import Dict
from curl_cffi import requests

from wela_agents.agents.llm import LLMAgent
from wela_agents.toolkit.toolkit import Tool
from wela_agents.models.openai_chat import OpenAIChat
from wela_agents.schema.template.openai_chat import ChatTemplate
from wela_agents.schema.template.openai_chat import UserMessageTemplate
from wela_agents.schema.template.openai_chat import SystemMessageTemplate
from wela_agents.schema.template.prompt_template import StringPromptTemplate

class WebBrowser(Tool):

    def __init__(self, model: OpenAIChat, proxies: Dict[str, Any] = None) -> None:
        super().__init__(
            name="visit_webpage",
            description="A web browser for visiting a specific URL.",
            required=["url", "question"],
            url={
                "type": "string",
                "description": "The url to visit.",
            },
            question={
                "type": "string",
                "description": "What question do you want to answer by browsing this webpage.",
            }
        )
        self.__model = model
        self.__proxies = proxies

    def _invoke(self, **kwargs: Any) -> str:
        try:
            url: str = kwargs["url"]
            question: str = kwargs["question"]

            html: str = requests.get(url, impersonate="chrome120", proxies=self.__proxies).content.decode(encoding="utf-8")
            content = trafilatura.extract(html)

            message_template_list = [
                UserMessageTemplate(
                    StringPromptTemplate(content)
                ),
                UserMessageTemplate(
                    StringPromptTemplate("Based on the above content, extract as comprehensively as possible the content that helps answer the following question: {question}")
                ),
                SystemMessageTemplate(
                    StringPromptTemplate("Output in the following format:\nContent_1\nContent_2\n...")
                )
            ]
            chat_template = ChatTemplate(message_template_list=message_template_list)
            agent = LLMAgent(model=self.__model, prompt_template=chat_template)
            ans = agent.predict(question=question)["content"]
            return ans

        except Exception as e:
            return f"{e}"

__all__ = [
    "WebBrowser"
]
