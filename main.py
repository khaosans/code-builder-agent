import os
import logging
import time  # Import time for rate limiting
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain.llms import OpenAI  # Import OpenAI model
import git  # Import gitpython for Git operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI model
llm = OpenAI(model="gpt-4")  # Use OpenAI's GPT-4 model

# Get the GITHUB_TOKEN from the environment variables (optional for authentication)
github_token = os.getenv('GITHUB_TOKEN')

if github_token:
    print("GITHUB_TOKEN retrieved successfully.")
else:
    print("GITHUB_TOKEN not found in environment variables.")

# Define the Git Status Tool
@tool("CheckGitStatusTool")
def check_git_status(repo_path: str) -> str:
    """Check the current status of the Git repository."""
    try:
        repo = git.Repo(repo_path)
        logger.info("Checking Git status.")
        changed_files = [item.a_path for item in repo.index.diff(None)]
        if changed_files:
            return f"Changed files: {changed_files}"
        else:
            return "No changes detected."
    except Exception as e:
        return f"Error checking Git status: {e}"

# Define the Git Structure Tool
@tool("CheckGitStructureTool")
def check_git_structure(repo_path: str) -> str:
    """Check the current structure of the Git repository."""
    try:
        repo = git.Repo(repo_path)
        logger.info("Checking Git repository structure.")

        # List all files and directories in the repo
        repo_files = [item.a_path for item in repo.tree().traverse()]
        if repo_files:
            return f"Repository structure: {repo_files}"
        else:
            return "Repository is empty."
    except Exception as e:
        return f"Error checking Git structure: {e}"

# Define the Code Builder agent using OpenAI
code_builder = Agent(
    role="Code Builder",
    goal="Build the Next.js application code and ensure it compiles successfully.",
    backstory="An expert build automation specialist using OpenAI, you ensure that the code is built efficiently and without errors.",
    verbose=True,
    llm=llm  # Assign OpenAI to this agent
)

# Define the task for checking Git status and code structure
git_structure_task = Task(
    description="Explore the Git repository and retrieve the code structure.",
    expected_output="A list of files and directories in the repository.",
    tools=[check_git_structure],  # Use the Git structure tool in this task
    agent=code_builder,
)

# Define the task for building the code
build_code_task = Task(
    description="Run the build command for the Next.js application and ensure there are no errors.",
    expected_output="The code should compile successfully without any errors.",
    agent=code_builder,
)

# Define a crew to handle both tasks with a sequential process
crew = Crew(
    agents=[code_builder],
    tasks=[git_structure_task, build_code_task],  # Run Git structure check first, then build
    process=Process.sequential,  # Tasks will run sequentially
)

# Kickoff inputs: Specify your build process and related information
kickoff_inputs = {
    'repo_path': '/Users/Sour/WebstormProjects/taskboard',  # Specify the path to your Git repository
    'build_command': 'npm run build',  # Example of a build command for Next.js
}

# Kickoff the process
result = crew.kickoff(inputs=kickoff_inputs)

# Introduce a rate limit between calls (optional)
time.sleep(2)  # Sleep for 2 seconds between tasks

# Output the result
print(result)
