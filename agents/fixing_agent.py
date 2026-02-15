import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FixingAgent:
    def __init__(self):
        # Load environment variables for OpenRouter
        self.api_key = os.getenv("OPENROUTER_API_KEY") 
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") # Default to OpenRouter URL

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        # Initialize OpenAI client pointing to OpenRouter
        # Consider a more powerful model for fixing tasks if needed (e.g., GPT-4o, Claude 3 Opus)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def fix_manim_code(self, failed_code: str, error_message: str) -> str | None:
        """
        Attempts to fix faulty Manim code based on an error message.

        Args:
            failed_code: The Manim Python code that caused the error.
            error_message: The stderr output from the failed Manim execution.

        Returns:
            A string containing the proposed fixed Manim code, or None on failure.
        """
        prompt = self._create_prompt(failed_code, error_message)
        
        # Prepare headers for OpenRouter
        extra_headers = None
        if self.base_url == 'https://openrouter.ai/api/v1':
             extra_headers = {
                "HTTP-Referer": "http://localhost:3000", # Replace with your actual site URL 
                "X-Title": "QuestionVideoExplainer"      # Replace with your actual site name
            }
        
        print("\n--- Attempting Code Fix ---")
        try:
            # Use a capable model for fixing code
            response = self.client.chat.completions.create(
                # model="openai/gpt-4o", # Recommended 
                model="openai/o3-mini-high", # Or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert Manim programmer. You will be given a Manim script that failed to render and the error message. Your task is to fix the script so it runs correctly. Output ONLY the complete, fixed Python code. Do not include explanations."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0, # Lower temperature for more focused fixes
                extra_headers=extra_headers
            )
            
            # Add robust checking for response structure
            if not response or not response.choices:
                print("Error fixing code: API response is missing choices.")
                return None
                
            choice = response.choices[0]
            if not choice.message or not choice.message.content:
                print(f"Error fixing code: API response choice is missing message content. Finish reason: {choice.finish_reason}")
                return None

            fixed_code = choice.message.content
            
            # Basic cleaning: remove potential markdown backticks
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code[len("```python"):].strip()
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-len("```")].strip()
                
            print("--- Proposed Fix Generated ---")
            # print(fixed_code) # Optional: Print the proposed fix
            return fixed_code

        except Exception as e:
            print(f"Error generating code fix: {e}")
            return None

    def _create_prompt(self, failed_code: str, error_message: str) -> str:
        """Helper function to create the prompt for the code fixing LLM."""
        
        # Limit error message length to avoid excessive prompt size
        max_error_len = 1500 
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n... [truncated] ..."

        prompt = f"""
        The following Manim script, which uses `manim-voiceover` with `GTTSService`, failed to render:
        ```python
        {failed_code}
        ```

        The error message produced was:
        ```
        {error_message}
        ```

        **Target Environment:** Manim Community **v0.18.0** with **manim-voiceover v0.3.7**.

        Please analyze the code and the error message, identify the problem (considering the target Manim version), and provide a fixed version of the complete Manim script. Think step-by-step to diagnose the issue before writing the fixed code.
        
        **Troubleshooting Steps & Analysis:**
        1.  **Traceback Analysis**: Carefully examine the traceback. Where does the error originate? Is it in the user's script (`generated_scene.py`) or deep within Manim or `manim-voiceover` libraries?
        2.  **Error Type**: What kind of error is it (`NameError`, `AttributeError`, `TypeError`, `ValueError`, `ImportError`)? What does the error message say specifically?
        3.  **Common Issues**: Check for:
            *   **Typos**: Misspelled variable names, function names, or keywords.
            *   **Imports**: Missing or incorrect imports.
            *   **Method Calls**: Incorrect arguments passed to Manim animations (`self.play`, `self.wait`) or other functions.
            *   **Object State**: Using Manim objects that haven't been created or added to the scene yet.
            *   **AttributeError Specifics**: If the error is an `AttributeError` (e.g., '`SomeMobject` object has no attribute `foo`'), **verify** if the method or attribute (`foo`) actually exists for that Manim object type (`SomeMobject`) based on standard Manim Community practices or documentation. The original code might be calling a non-existent method (like `.lower()` on a Mobject). If so, correct the code to use a valid method or achieve the intended behavior differently (e.g., change `self.add()` order for layering).
            
                **Common Camera `AttributeError`s:**
                *   **Error:** `AttributeError: 'Camera' object has no attribute 'animate'`. **Diagnosis:** This likely means the code is incorrectly trying `self.camera.animate...` instead of animating the camera's frame.
                *   **Error:** `AttributeError: 'Camera' object has no attribute 'frame'`. **Diagnosis:** This error is unexpected, as camera animations typically use `self.camera.frame`. Check if `self.camera` was somehow overwritten or if the Scene type has an unusual camera setup. However, the goal is usually to animate the frame.
                *   **Correct Usage (Based on Documentation for Manim v0.18.0):** Camera animations (like zooming, panning, changing width) target the camera's frame (`self.camera.frame`) and MUST be wrapped in `self.play()`. The correct pattern is `self.play(self.camera.frame.animate.[method/property](...))`. 
                    *   **Setting Width:** `self.play(self.camera.frame.animate.set(width=target_width))`
                    *   **Moving Center:** `self.play(self.camera.frame.animate.move_to(target_mobject_or_point))`
                *   **Scaling (Zooming):** `self.play(self.camera.frame.animate.scale(scale_factor))`
                *   **Chaining:** `self.play(self.camera.frame.animate.move_to(target).set(width=new_width))`
                *   **Saving/Restoring:** Use `self.camera.frame.save_state()` before the animation and `self.play(Restore(self.camera.frame))` afterwards if needed.
                **Fix the code to use these documented `self.play(self.camera.frame.animate...)` patterns.**
            *   **Voiceover Issues**: 
                *   **`construct` Method**: Ensure `construct` is **synchronous** (`def construct(self):`), NOT `async def`.
                *   **Service Initialization**: Check if `self.set_speech_service(GTTSService())` is called correctly at the beginning of `construct`.
                *   **`voiceover` Arguments**: If the error occurs within `self.voiceover(...)` or the TTS service calls (like in `manim_voiceover/services/base.py` or `gtts.py`), **verify the keyword arguments passed to `self.voiceover`**. For `GTTSService`, the language argument MUST be `lang="..."`, NOT `language="..."`. Check for other invalid arguments.
        4.  **Logic Errors**: Review the animation logic within the failing scene or `with` block. Does the sequence make sense?

        **Fixing Guidelines:**
        *   **Address the Root Cause**: Make sure your fix directly addresses the specific error identified.
        *   **Minimal Changes**: Only change the necessary parts of the code. Preserve working code, comments, and the overall structure, especially the voiceover blocks.
        *   **Complete Script**: Output the ENTIRE fixed Python script.
        *   **Imports & Class Structure**: Ensure the final script includes all necessary imports (`from manim import *`, `from manim_voiceover import VoiceoverScene`, `from manim_voiceover.services.gtts import GTTSService`, `import numpy as np`), defines the `GeneratedVideoScene(VoiceoverScene)` class, and has the correct synchronous `def construct(self):` method for Manim v0.18.0.
        *   **Output Format**: Output ONLY the corrected, complete Python code. Do not add explanations or markdown formatting.

        Provide the fixed Python code:
        """
        return prompt
