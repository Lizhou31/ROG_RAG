from typing import Dict, Any
from langchain.prompts import PromptTemplate
try:
    from LLM_Provider import BaseLLM, OpenAILLM, OllamaLLM
except ImportError:
    from Utility.LLM_Provider import BaseLLM, OpenAILLM, OllamaLLM

class DocGrader:
    def __init__(self):
        self.grading_prompt = self._create_grading_prompt()

    def _create_grading_prompt(self) -> PromptTemplate:
        template = """
        You are an expert at evaluating document retrieval results. It is important that the products name should be in the retrieved docs.
        Please analyze the following retrieval results and grade them based on:
        Relevance: How relevant are the retrieved docs to the query (0-10)
        
        Query:
        - Topic: {topic}
        - Description: {description}

        Retrieved Results:
        - Retrieved Docs: {retrieved_docs}

        Please output only in JSON format with no other text:
        "relevance":[score1, score2, score3...]
        """
        return PromptTemplate.from_template(template)

    def grade_retrieval(self, 
                       topic: str,
                       description: str, 
                       retrieval_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Grade the retrieval results using the specified LLM provider.
        
        Args:
            topic: The original query topic
            description: The original query description
            retrieval_result: Dictionary containing retrieved docs
        
        Returns:
            Dictionary containing relevance score
        """
        input_data = {
            "topic": topic,
            "description": description,
            "retrieved_docs": retrieval_result
        }

        formatted_input = self.grading_prompt.format(**input_data)
        llm = OpenAILLM().get_llm()
        result = llm.invoke(formatted_input)
        
        return OpenAILLM().process_result(result)
    
if __name__ == "__main__":
    grader = DocGrader()
    result = grader.grade_retrieval("My Logitech G502 Hero is not working", 
                                    "I recently bought a Logitech G502 Hero and it is not working. I have tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working. I have also tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working. I have also tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working.",
                                    "Product: ROG Chakram X\nDescription: ROG Chakram X wireless RGB gaming mouse with next-gen 36,000 dpi ROG AimPoint optical sensor, 8000 Hz polling rate, low-latency tri-mode connectivity (RF 2.4 GHz / Bluetooth / wired), 11 programmable buttons, an analog joystick and hot-swappable micro switch sockets (mechanical / optical ).\
                                    Product: ROG Spatha X\nDescription: Wireless gaming mouse with dual-mode connectivity (wired / 2.4 GHz), magnetic charging stand, 12 programmable buttons, ROG 19,000 dpi sensor, Exclusive Push-Fit Switch Sockets, ROG Micro Switches, ROG Paracord, Aura Sync RGB lighting.\
                                    Product: ROG Gladius III Wireless AimPoint\nDescription: The ROG Gladius III Wireless AimPoint is a lightweight 79-gram wireless RGB gaming mouse that features a 36,000-dpi ROG AimPoint optical sensor, tri-mode connectivity, ROG SpeedNova wireless technology, swappable mouse switches, ROG Micro Switches, pivoted button mechanism for 0 ms click latency, ergonomic design, ROG Paracord, 100% PTFE mouse feet, six programmable buttons, and mouse grip tape.")
    print(result)