# -----
# Step 1 initiate the model
# -----
import math
from langchain.agents import create_agent
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:
    raise ImportError(
        "Missing langchain-google-genai; install it with "
        "`pip install langchain-google-genai`."
    ) from exc
from langchain_core.tools import tool
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    #temperature=0
)

# ----
# defining tool
# ----


@tool
def add(a: float, b: float) -> float:
    """Add two number, use for addition operation"""
    return a+b


@tool
def multiply(a: float, b: float) -> float:
    """multiply two number, use for multiplecation operation"""
    return a*b


@tool
def divide(a: float, b: float) -> float:
    """divide two number, use for division operation"""
    return a/b


@tool
def square_root(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: cannot take square root of a nagative number"
    return math.sqrt(number)


tools = [add, multiply, divide, square_root]

agent = create_agent(
    model=model, tools=tools
)


def run_agent(question: str):
    print(f"user: {question}")
    print("-" * 50)

    result = agent.invoke(
        {
            "messages": [("user", question)]
        }
    )

    final_message = result["messages"][-1]
    print("agent:", final_message.text)

#simple
run_agent("What is the sum of 5 and 10?")
#medium
run_agent("What is 8 multiplied by 3, then divided by 4?")

#complex
run_agent("I have a rectangle with a length of 12 units and a width of 5 units. What is the area of the rectangle, and what is the square root of that area?")
run_agent("Who is president of India?")