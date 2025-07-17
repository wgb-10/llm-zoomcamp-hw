from IPython.display import display, HTML
import markdown

from google.genai import types


class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        self.tools[function.__name__] = types.Tool(function_declarations=[description])

        self.functions[function.__name__] = function

    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        function_name = tool_call_response.name
        arguments = tool_call_response.args

        f = self.functions[function_name]
        result = f(**arguments)

        return types.Part.from_function_response(
            name=function_name,
            response={"result": result},
        )


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


class ChatInterface:
    def input(self):
        question = input("You:")
        return question

    def display(self, message):
        print(message)

    def display_function_call(self, entry, result):
        call_html = f"""
            <details>
            <summary>Function call: <tt>{entry.name}({shorten(entry.args)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result.response['result']}</pre>
            </div>
            
            </details>
        """
        display(HTML(call_html))

    def display_response(self, entry):
        response_html = markdown.markdown(entry.text)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html))


class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client):
        self.developer_prompt = developer_prompt
        self.tools = tools
        self.chat_interface = chat_interface
        self.client = client

    def gpt(self, chat_messages):
        return self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=chat_messages,
            config=types.GenerateContentConfig(
                tools=self.tools.get_tools(),
                system_instruction=self.developer_prompt,
                temperature=0,  # setting to 0 (for reliable function calls)
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),  # Disables thinking
            ),
        )

    def run(self):

        chat_messages = []

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            chat_messages.append(
                types.Content(role="user", parts=[types.Part(text=question)])
            )

            while True:  # inner request loop
                response = self.gpt(chat_messages)

                has_messages = False

                for entry in response.candidates[0].content.parts:
                    chat_messages.append(entry)

                    if entry.function_call:
                        result = self.tools.function_call(entry.function_call)
                        chat_messages.append(types.Content(role="user", parts=[result]))
                        self.chat_interface.display_function_call(
                            entry.function_call, result.function_response
                        )

                    else:
                        self.chat_interface.display_response(entry)
                        has_messages = True

                if has_messages:
                    break
