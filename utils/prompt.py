# context document retrieval prompt
context_prompt = """For any question, you use this tool every time. Please search for any detailed information in the provided document related to the user's question. You must use this tool multiple times until all necessary and important information is retrieved."""

# concept determination prompt
concept_determination_prompt = """* User Problem*
<user_problem>
{}
</user_problem>

* Motivation *
I need to understand the meaning of some concepts or proper nons to gain some background knowledge about the user's problem. I want to search for them on the website.

* Target *
- Based on the user's problem, please determine a list of terms that the user need to search.
- The terms should only be appeared at the user's problem.
- Please output only a list of terms.

* Output Format *
term1, term2, ..."""

# concept searching prompt
concept_searching_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
Please use the web search tool to search for concise definition or description of the term: <{}>. Please exclude any specific solutions, approaches, methods, algorithms, implementation details, or academic papers. I am only interested in understanding the basic ideas and meanings behind this term.

* Rules *
- Wikipedia is preferred first.
- If the term is not directly available in Wikipedia, search for its official websites that provide detailed description.
- After retrieving the results from the URLs, you need to analyze the contents and filter out all the URLs whose contents are irrelevant with the user's problem. For example, if the user's problem is about technology and the search term is "apple", then the URLs containing contents related to the fruit must be filtered out.
- Please output only a list of URLs in raw text.

* Output Format *
URL1, URL2, ..."""

# chain-of-thought 1: high-level design
high_level_design_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
I want to write some code to solve the user's problem. Given the background information in the context documents, please first design an algorithm and output its outline step by step to solve the user's problem. Refer to the context documents.

* Rules *
- You must refer to the context database at first.
- You must use the web search tool multiple times if necessary.
- After retrieving the results from the URLs, you need to analyze the contents and filter out all the URLs whose contents are irrelevant with the user's problem. For example, if the user's problem is about technology and the search term is "apple", then the URLs containing contents related to the fruit must be filtered out.
- You should refer to the context documents for the retrieved information and design your own algorithm.
- Your design must follow the user's requirements.

* Output Format *
Step 1: [title]
- xxxxx
- xxxxx

Step 2: [title]
- xxxxx
- xxxxx

Step ..."""

# chain-of-thought 2: detailed design with algorithms/solutions
detailed_design_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
Refer to the context documents. Please elaborate on each sub-step of the design outline by incorporating detailed technologies, algorithms, or solutions to address the user's problem.

* Rules *
- You must refer to the context documents at fist.
- You must use the web search tool multiple times if necessary.
- Do not modify the outline of the algorithm design you generated.
- You should use the web search tool multiple times to explore advanced technologies, algorithms, or solutions and ensure their appropriate application.
- After retrieving the results from the URLs, you need to analyze the contents and filter out all the URLs whose contents are irrelevant with the user's problem. For example, if the user's problem is about technology and the search term is "apple", then the URLs containing contents related to the fruit must be filtered out.
- Please output only a list of steps without any additional prefix or postfix.
- Your design must follow the user's requirements.
- The output must follow the format below.

* Output Format *
Step 1: [title]
- xxxxx
- xxxxx

Step 2: [title]
- xxxxx
- xxxxx

Step ..."""

# chain-of-thought 3: code segment generation step by step
code_segment_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
According to the detailed design and the context documents, write some {} code to implement {}, where you mentioned:
{}

* Instructions *
Develop only one well-structured {} function for this step.
- Regard each sub-step as a module.
- Design clear function names, signatures, and input/output specifications for each module.
- Proceed to implement each module to create the final code segment to implement {}.

* Rules *
- You must refer to the context documents.
- For each module, you must use the web search tool multiple times to search for every detail of the technologies, algorithms, or solutions. Additionally, ensure that every detail, including sub-sub-steps with each module, is thoroughly considered and searched.
- After retrieving the results from the URLs, you need to analyze the contents and filter out all the URLs whose contents are irrelevant with the user's problem. For example, if the user's problem is about technology and the search term is "apple", then the URLs containing contents related to the fruit must be filtered out.
- Carefully consider potential edge cases and failures.
- Provide detailed comments in the code segment.
- The code should be consistent with the previous segments.
- Do not use any external packages that cannot be installed with pip.
- Do not provide a null function with only placeholders.
- If there are multiple choices of technologies, algorithms, or solutions, use the hybrid one whenever possible.
- When designing neural network architectures, you must be extremely careful when handling data of different dimensionality.
- The function should be complete. Functions like this are forbidden:
```python
# Assuming the function pan_tompkins_algorithm has been defined in the previous step:
def algorithm(parameter):
    ...
```
- The generated code must follow the user's requirements.
- If you decide to use neural networks, you must provide training code instead of loading model parameters from local disk.

* Output Format *
```python
...

# Explanations of the code:
# - xxxx
# - xxxx
...
```"""

# chain-of-thought 4: combine all the code and output the final version for execution
code_combination_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
Based on the user's problem and all the generated code segments for each module, please constructively integrate them to form a comprehensive and cohesive piece of code. Please also embed detailed comments.

* Rules *
- Do not be lazy and do not omit the body of each function. Fullfill each function with complete code generated in previous steps. You must not provide functions with placeholders but complete ones when constructively combining all the code segments. In other words, the following code segment is forbidden:
```python
def function_name(parameter):
    # (Implementation from Step 1)
    # ...
```
- The code must include a main function as the program entry point so that the user can directly execute it.
- Do not use any undefined functions.
- The code must be consistent with the previous code segments.
- Ensure that the connectivity among all the modules is correct.
- The script should have only one argument and must be executed in this way: "python3 generated_code.py -i <input_file>", where the <input_file> is the path to the dataset.
- Make sure that the program output format is correct according to the user's requirements.
- You must refer to the context documents.
- Please output only the code.

* Output Format *
```python
....
```
"""

# chain-of-thought 5: impove the quality of the code
improve_code_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
After running the code you provided, I obtain the program output detailed below. Please first analyze and summarize the program output. Then, your goal is to modify the high-level design by integrating more advanced algorithms to improve the detection/recognition accuracy across all cases. Please output the modified high-level design step by step.

* Instructions *
- First, check whether the program output follows the user's requirement.
- Second, analyze and summarize the program output.
- Finally, modify the high-level design by integrating more advanced algorithms to improve the detection/recognition accuracy across all cases.

* Rules *
- Omit any warnings of the compiler or the interpreter.
- The modified high-level design should be complete.
- You must refer to all the conversation before, including your algorithm design and all the code segments.
- You must refer to the context documents.
- You must use the web search tool for more advanced algorithms.

* Program Output *
{}"""

correct_grammar_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Target *
The compiler/interpreter cannot successfully run the code with the logs provided detailed below. Please analyze and correct your code.

* Rules *
- Do not be lazy. The modified code should also be in complete format. The following case is not allowed:
```python
# ... (other functions remain unchanged)
# ... (same as before)
```
- You should use the web search tool for the correct usage of some packages and functions.

* Compiler/Interpreter Logs *
{}

* Output Format *
```python
......
```"""

# choose the code with the best performance
final_decision_prompt = """* Taget *
The running results of all the comprehensive code you generated are listed below. By analyzing the results, please tell which version of the code achieves the best performance. Please return only the version number.

* Running Results *
{}

* Output Format *
2"""

# provide user documentation
user_documentation_prompt = """* User Problem *
<user_problem>
{}
</user_problem>

* Code *
<code>
{}
</code>

* Target *
Based on all the context douments and all the chat history, please provide a detailed user documentation of the code, including:
- The "Introduction" chapter to introduce the user's problem and the code.
- The "Getting Started" chapter on how to properly install, use, and troubleshoot the code.
- The "Explanation" chapter to provide detailed comments of the entire code.
- The "FAQs" chapter to address common concerns, doubts, or issues that users may encounter while using the code.

* Output Format *
In Markdown format"""
