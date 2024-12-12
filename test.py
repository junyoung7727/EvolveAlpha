import os
import json
import logging
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from tavily import TavilyClient

# 환경 변수 로드
load_dotenv()

############################################
# WebSearchAgent: Handles web search
############################################
class WebSearchAgent:
    """
    Handles web search operations using an external API.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger("WebSearchAgent")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = OpenAI(api_key=self.api_key)


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search.

        Args:
            query (str): The search query.

        Returns:
            Dict[str, Any]: Search results or an error message.
        """
        try:
            # TavilyClient는 가상의 API 클라이언트로, 필요시 교체해야 함
            response = TavilyClient(api_key=self.api_key).search(query=query)
            self.logger.info(f"Web search successful for query: {query}")
            return {"status": "success", "results": response}
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return {"status": "error", "error": str(e)}

############################################
# LLMClient: Manages LLM calls
############################################
class LLMClient:
    """
    Manages interactions with the OpenAI GPT-4 API for reasoning tasks.
    """
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = logging.getLogger("LLMClient")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Call the OpenAI LLM.

        Args:
            messages (List[Dict[str, str]]): Input messages for the LLM.
            temperature (float): Sampling temperature for output variability.

        Returns:
            str: LLM's response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content.strip()
            self.logger.info("LLM response received successfully.")
            return content
        except Exception as e:
            self.logger.error(f"LLM call error: {str(e)}")
            raise

    def decide_need_web_search(self, question: str) -> bool:
        """
        Determine if web search is needed to answer the question.

        Args:
            question (str): The user's question.

        Returns:
            bool: True if web search is needed, otherwise False.
        """
        system_prompt = """\
You are a highly logical and reasoning-focused AI model.
Decide if the following question requires external information to be answered completely and accurately.
Respond with either "NEED_WEB_SEARCH" or "NO_NEED_WEB_SEARCH".
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        output = self._call_llm(messages)
        return "NEED_WEB_SEARCH" in output

    def create_plan(self, question: str, web_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a reasoning plan based on the question and optional web results.

        Args:
            question (str): The user's question.
            web_results (Optional[Dict[str, Any]]): Search results if a web search was performed.

        Returns:
            Dict[str, Any]: The reasoning plan and instruction.
        """
        system_prompt = """\
You are a state-of-the-art AI model.
Create a reasoning plan for the given question. Optionally integrate web search results if provided.

Output only valid JSON:
{
  "plan": [...],  # List of logical reasoning steps.
  "instruct": "..."  # Single next-step instruction.
}
"""
        web_data = json.dumps(web_results, ensure_ascii=False) if web_results else "null"
        user_content = f"Question: {question}\nWeb results: {web_data}\nCreate a plan and next instruction."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        output = self._call_llm(messages)
        return json.loads(output)

    def execute_step(self, plan: List[str], step: str, previous_output: str) -> str:
        """
        Execute a single step in the reasoning process.

        Args:
            plan (List[str]): The reasoning plan.
            step (str): The current step to execute.
            previous_output (str): Result from the previous step.

        Returns:
            str: Result of the current step execution.
        """
        system_prompt = f"""\
You are executing a reasoning step in a multi-step plan.
Current step: {step}
Previous results: {previous_output}

Rules:
1. Produce a concise and logical conclusion starting with "O1:".
2. Ensure clarity, correctness, and logical consistency.
"""
        user_content = f"Execute the step: {step}\nPlan: {json.dumps(plan)}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        output = self._call_llm(messages)
        return output.strip()

############################################
# ProcessPipeline: Handles the reasoning workflow
############################################
class ProcessPipeline:
    """
    Manages the entire reasoning process, integrating LLM interactions and web search.
    """
    def __init__(self, llm_client: LLMClient, web_agent: WebSearchAgent):
        self.llm_client = llm_client
        self.web_agent = web_agent
        self.logger = logging.getLogger("ProcessPipeline")

    def run(self, question: str) -> str:
        """
        Run the reasoning pipeline for the given question.

        Args:
            question (str): The user's question.

        Returns:
            str: The final answer.
        """
        self.logger.info("Starting the reasoning pipeline.")
        need_web = self.llm_client.decide_need_web_search(question)
        web_results = self.web_agent.search(question) if need_web else None

        plan_data = self.llm_client.create_plan(question, web_results)
        plan = plan_data["plan"]
        instruct = plan_data["instruct"]
        self.logger.info(f"Plan created with {len(plan)} steps.")

        step_results = []
        for step in plan:
            prev_result = step_results[-1] if step_results else ""
            step_result = self.llm_client.execute_step(plan, step, prev_result)
            step_results.append(step_result)

        final_answer = step_results[-1] if step_results else "No steps executed."
        self.logger.info("Pipeline completed successfully.")
        return final_answer

############################################
# Main function
############################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    llm_client = LLMClient(model_name="gpt-4o-mini")
    web_agent = WebSearchAgent()
    pipeline = ProcessPipeline(llm_client, web_agent)

    problem_text = """
    최고차항의 계수가 1인 삼차함수 f(x)에 대하여 함수 g(x)를
    g(x) = f(e^x) + e^x
    이라 하자. 곡선 y = g(x) 위의 점 (0, g(0))에서의 접선이 x축이고 함수 g(x)가 역함수 h(x)를 가진다.
    이때 h'(8)의 값을 구하시오.
    """
    final_answer = pipeline.run(problem_text)
    print("Final Answer:", final_answer)
