import asyncio
from typing import List, Dict
import os
import subprocess
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.tasks import Task
from camel.workforce import Workforce
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.base import BaseToolkit
from camel.toolkits.openai_function import OpenAIFunction


class QuarkusBean:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path).split('.')[0]
        self.content, self.doc_comment = self.read_file()

    def read_file(self):
        with open(self.file_path, 'r') as file:
            content = file.read()
            doc_comment = self.extract_doc_comment(content)
        return content, doc_comment

    def extract_doc_comment(self, content):
        start = content.find("/**")
        end = content.find("*/", start)
        if start != -1 and end != -1:
            return content[start:end + 2].strip()
        return ""


class EncaptWorkforce(BaseToolkit):
    PROJECT_ROOT = "./encapt-project"
    BEANS_DIR = os.path.join(PROJECT_ROOT, "src", "main", "kotlin", "org", "camelai", "beans")

    def __init__(self):
        self.workforce = Workforce('Quarkus Codebase Manager')
        self.beans = self._load_beans()
        self._setup_workforce()

    def _load_beans(self) -> List[QuarkusBean]:
        beans = []
        for root, dirs, files in os.walk(self.BEANS_DIR):
            for file in files:
                if file.endswith(".kt"):
                    file_path = os.path.join(root, file)
                    beans.append(QuarkusBean(file_path))
        return beans

    def _setup_workforce(self):
        function_list = self.get_tools()

        model_config = ChatGPTConfig(
            tools=function_list,
            temperature=0.0,
        )

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=model_config.as_dict(),
        )

        # Create manager agent
        manager_sysmsg = BaseMessage.make_assistant_message(
            role_name="Manager Agent",
            content="""You are responsible for managing the Quarkus codebase.
            Your responsibilities include:
            1. Coordinating changes across multiple beans
            2. Ensuring overall project consistency
            3. Delegating tasks to appropriate bean agents

            Use the provided tools to write files, run tests, get test summaries, and send messages to other agents when needed.

            When creating a new agent or bean, only write a kotlin class with appropriate Quarkus annotations and a doc comment that outlines the responsibilities of the bean. The agent will handle the rest.

            Always wrap your final response in <result> tags.
            """
        )
        self.workforce.add_single_agent_worker(
            "Manager",
            ChatAgent(manager_sysmsg, model=model, tools=function_list)
        )

        # Create bean agents
        for bean in self.beans:
            self._create_bean_agent(bean, model, function_list)

    def _create_bean_agent(self, bean: QuarkusBean, model, function_list):
        sysmsg = BaseMessage.make_assistant_message(
            role_name=f"{bean.name} Agent",
            content=f"""You are responsible for the {bean.name} Quarkus bean.
            Your responsibilities include:
            {bean.doc_comment}

            Here are the other beans and their responsibilities, derived from the classes doccomment:
            {', '.join([f'{b.name}: {b.doc_comment}' for b in self.beans if b.name != bean.name])}. Inject them using quarkus @Inject annotation as appropriate and needed.

            Fully implement the responsibilities of the bean.
            Use the provided tools to read and modify your bean's content, run tests, get test summaries, and send messages to other agents.
            When you notice code errors or a failed test, edit the code and try running again, until the tests pass.
            Coordinate with the Manager Agent for changes that affect multiple beans. When you make changes, ensure they are tested and working correctly. Edit and test until you are happy.

            Message another agent with your message tool if you need to know more details about its service, ask for help, or request a change. You should also work with them to meet the shared responsibilities of the beans.

            Always wrap your final response in <result> tags.
            """
        )
        self.workforce.add_single_agent_worker(
            bean.name,
            ChatAgent(sysmsg, model=model, tools=function_list)
        )

    def write_file(self, file_name: str, content: str) -> str:
        """
        Writes content to a Kotlin file in the beans directory.

        Args:
            file_name (str): The name of the file to write. Must end with '.kt'.
            content (str): The content to write to the file.

        Returns:
            str: A message indicating success or describing an error if one occurred.
        """
        if not file_name.endswith('.kt'):
            return "Error: Only .kt files are allowed."

        try:
            file_path = os.path.join(self.BEANS_DIR, file_name)
            with open(file_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote content to {file_name}"
        except IOError as e:
            return f"Error writing to file: {str(e)}"

    def create_directory(self, dir_name: str) -> str:
        """
        Creates a new directory within the beans directory.

        Args:
            dir_name (str): The name of the directory to create.

        Returns:
            str: A message indicating success or describing an error if one occurred.
        """
        try:
            dir_path = os.path.join(self.BEANS_DIR, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            return f"Successfully created directory {dir_name}"
        except OSError as e:
            return f"Error creating directory: {str(e)}"

    def run_all_bean_tests(self) -> str:
        """
        Runs all tests for beans in the Quarkus project.

        Returns:
            str: The output of the test execution or an error message if the tests fail.
        """
        try:
            result = subprocess.run(
                ['./gradlew', 'test', '--tests=org.camelai.beans.*'],
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running bean tests: {e.output}"

    def run_specific_bean_test(self, bean_name: str) -> str:
        """
        Runs tests for a specific bean in the Quarkus project.

        Args:
            bean_name (str): The name of the bean to test (without 'Test' suffix).

        Returns:
            str: The output of the test execution or an error message if the test fails.
        """
        try:
            result = subprocess.run(
                ['./gradlew', 'test', f'--tests=org.camelai.beans.{bean_name}Test'],
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running test for {bean_name}: {e.output}"

    def send_message(self, to_agent: str, from_agent: str, message: str) -> str:
        """
        Sends a message to another agent by creating a task for them using the workforce's process_task method.

        Args:
            to_agent (str): The name of the agent to send the message to.
            from_agent (str): The name of the agent sending the message.
            message (str): The content of the message to send.


        Returns:
            str: A confirmation message indicating that the task was created for the recipient agent.
        """
        task = Task(
            content=f"""You have received a message from {from_agent}:

{message}

Please process this message and respond appropriately. If any actions are required, take them using your available tools.

Use the message tool to respond to {from_agent} with your response. Always respond something, but fulfill the request first if possible.

Always put <response> tags around your response. For example:
<response>I have processed your request and updated the UserService.</response>
""",
            id=f'inter_agent_message_{to_agent}',
        )

        asyncio.run(self.workforce._channel.post_task(task, from_agent, to_agent))
        return f"Message sent to {to_agent}. A task has been created for them to respond."

    def get_tools(self) -> List[OpenAIFunction]:
        """
        Returns a list of all available tools as OpenAIFunction objects.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects representing all available tools.
        """
        return [
            OpenAIFunction(self.write_file),
            OpenAIFunction(self.create_directory),
            OpenAIFunction(self.run_all_bean_tests),
            OpenAIFunction(self.run_specific_bean_test),
            OpenAIFunction(self.send_message),
        ]

    def process_change_request(self, request: str) -> Task:
        """
        Processes a change request by creating and executing a task in the workforce.

        Args:
            request (str): The change request to process.

        Returns:
            Task: The completed task object containing the result of processing the change request.
        """
        task = Task(
            content=f"""Process the following change request:

    {request}

    Analyze which beans need to be modified or created. Use the provided tools to make necessary changes,
    run tests, and ensure the changes are working correctly. Coordinate between agents as needed. Ensure a bean is only edited by its agent. If you believe a change needs to be made, communicate with that agent using the send_message tool to request the change.

    Always put <result> tags around the result of your actions. For example, <result>Modified UserService to include a new method.</result>. Include ALL relevant details in the result tag, multiple lines are fine, but there must only be one tag.

""",
            id='change_request',
        )
        result = self.workforce.process_task(task)
        return result


def main():
    quarkus_workforce = EncaptWorkforce()

    # Example: Request a change
    result = quarkus_workforce.process_change_request("""
    Implement a new feature: Add a method in UserService to retrieve the user's order history,
    and create a new OrderHistoryService to handle the retrieval of order history.
    Ensure proper error handling and update the documentation accordingly.
    Then, run tests to verify the changes.
    """)

    print("Change request result:", result.content)


if __name__ == "__main__":
    main()
