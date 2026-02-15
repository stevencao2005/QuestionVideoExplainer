# Question Video Explainer

A project to automatically generate explanatory videos from questions using Manim and AI agents.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd QuestionVideoExplainer
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Keys:**
    Create a `.env` file in the root directory:
    ```
    OPENROUTER_API_KEY='your_openrouter_key'
    OPENAI_BASE_URL='https://openrouter.ai/api/v1' # Keep this line for OpenRouter
    ```

## Usage

Run the main script with a question:

```bash
python main.py --question "Explain the Pythagorean theorem."
```

*(More detailed usage instructions to come as the project develops)* 