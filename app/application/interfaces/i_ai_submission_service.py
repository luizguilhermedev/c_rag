from abc import ABC, abstractmethod


class IAISubmissionService(ABC):
    @abstractmethod
    def process_message(self, input_message):
        """
        Process a message through the graph and stream responses.

        Args:
            input_message (str): User input message

        Returns:
            list: List of response messages
        """
        pass
