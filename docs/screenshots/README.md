# Screenshots

`project.md` asks for screenshots, and they are the fastest way for a reviewer to
see that the project works. Drop the images in this folder and reference them
from the README.

Worth capturing, in this order of value:

1. **`chat-answer.png`** — the chat app having answered a question, with the
   Sources expander open so the citations and links are visible. Pick a question
   whose answer shows the model citing `[1]`, `[2]` in the text.
2. **`dashboard.png`** — the dashboard with real traffic on it. Ask a dozen
   questions and leave some thumbs first, otherwise most charts render an empty
   notice. Turn on "Score answers with the LLM judge" in the chat sidebar for a
   few of them so charts 5 and 6 have data.
3. **`retrieval-eval.png`** — the terminal output of `make eval-retrieval`,
   showing the comparison table. Optional, since the numbers are in the README
   and the CSV, but it makes the evaluation concrete.
4. **`no-context.png`** — the assistant declining to answer something outside the
   corpus. This is worth showing precisely because it is the behaviour the
   grounded prompt exists to produce.

A short screen recording works too. Streamlit can record one from the menu in the
top-right of the app; drag the file into the GitHub editor for the README and it
will upload and embed it.
