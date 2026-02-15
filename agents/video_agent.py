import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VideoAgent:
    def __init__(self):
        # Load environment variables for OpenRouter
        self.api_key = os.getenv("OPENROUTER_API_KEY") 
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") # Default to OpenRouter URL

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        # Initialize OpenAI client pointing to OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate_manim_code(self, plan: dict) -> str | None:
        """
        Generates Manim code from a structured video plan.

        Args:
            plan: The structured video plan (dictionary).

        Returns:
            A string containing the generated Manim Python code, or None on failure.
        """
        prompt = self._create_prompt(plan)
        
        # Prepare headers for OpenRouter
        extra_headers = None
        if self.base_url == 'https://openrouter.ai/api/v1':
             extra_headers = {
                "HTTP-Referer": "http://localhost:3000", # Replace with your actual site URL 
                "X-Title": "QuestionVideoExplainer"      # Replace with your actual site name
            }

        try:
            # Using a capable model is recommended for code generation
            response = self.client.chat.completions.create(
                model="openai/o3-mini-high",
                messages=[
                    {"role": "system", "content": "You are an expert Manim programmer. Generate Python code for a Manim animation with voiceover based on the provided plan. Output ONLY the Python code."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5, # Lower temperature for more deterministic code
                extra_headers=extra_headers
            )
            
            # --- Add robust checking for response structure ---
            if not response or not response.choices:
                print("Error generating Manim code: API response is missing choices.")
                # Log the full response if possible (might be large)
                # print(f"Full API Response: {response}") 
                return None
                
            choice = response.choices[0]
            if not choice.message or not choice.message.content:
                print(f"Error generating Manim code: API response choice is missing message content. Finish reason: {choice.finish_reason}")
                return None
            # --- End checking ---

            manim_code = choice.message.content
            
            # Basic cleaning: remove potential markdown backticks
            if manim_code.startswith("```python"):
                manim_code = manim_code[len("```python"):].strip()
            if manim_code.endswith("```"):
                manim_code = manim_code[:-len("```")].strip()
                
            print("--- Generated Manim Code ---")
            print(manim_code)
            print("-----------------------------")
            return manim_code

        except Exception as e:
            print(f"Error generating Manim code: {e}")
            return None

    def _create_prompt(self, plan: dict) -> str:
        """Helper function to create the prompt for the Manim code generation LLM."""
        
        plan_str = json.dumps(plan, indent=2)

        prompt = f"""
        You are an expert Manim Community Edition developer. Generate a complete, single Python script using Manim and the manim-voiceover library to create an animation based on the following plan:

        **Plan:**
        ```json
        {plan_str}
        ```

        **Code Generation Guidelines:**
        1.  **Imports:** Include ALL necessary imports explicitly at the top (`from manim import *`, `from manim_voiceover import VoiceoverScene`, `from manim_voiceover.services.gtts import GTTSService`, plus any other required Manim modules like `numpy`).
        2.  **Speech Service:** Use `GTTSService`. Initialize it within `construct` using `self.set_speech_service(GTTSService())`.
        3.  **Scene Class:** Create a single Python class named `GeneratedVideoScene` that inherits from `VoiceoverScene`.
        4.  **Structure:** Implement the animation logic within the **synchronous** `def construct(self):` method. Do NOT make it `async def`. Consider using helper functions within the class for complex or repeated animation sequences if it improves clarity and modularity, but ensure all code is within the single script.
        5.  **Plan Adherence:** Strictly follow the sequence of scenes, narration, and animation descriptions provided in the JSON plan.
        6.  **Voiceover Synchronization:** Use `with self.voiceover(text=narration, lang="en") as tracker:` blocks for each scene, using the exact `narration` text from the plan. **IMPORTANT: Use the `lang` argument (e.g., `lang="en"`) for specifying the language, NOT `language`.** Translate the natural language `animations` from the plan into Manim API calls within the corresponding `with` block.
        7.  **Code Quality:** Write clear, efficient, and well-commented code following Manim Community Edition best practices. Add comments to explain non-obvious logic or complex transformations.
        8.  **Layout Management:** **CRITICAL**: Pay close attention to the layout hints in the plan. Use Manim's layout helpers whenever possible to achieve the specified positioning and avoid overlaps:
            *   Group related objects using `VGroup()`.
            *   Use `group.arrange(DIRECTION, buff=VALUE)` to arrange items in a VGroup.
            *   Use `object2.next_to(object1, DIRECTION, buff=VALUE)` for precise relative positioning.
            *   Use `object.move_to(TARGET_POINT_OR_OBJECT)` for absolute or relative centering.
            *   Carefully manage layers using `self.add()` order or `z_index` if necessary.
            *   **PRIORITIZE AVOIDING OVERLAPS** between unrelated elements.
        9.  **API Accuracy:** **CRITICAL**: Only use methods and attributes that actually exist for Manim objects (Mobjects, Scenes, etc.) according to the official Manim Community documentation. Do NOT invent methods (like `.lower()`). Verify method calls and parameters. For layering, use the order of `self.add()` or standard positioning methods (`.move_to()`, `.shift()`, etc.). **Specifically, when animating the camera, use `self.camera.frame.animate...`, NOT `self.camera.animate...`.**
        10. **No Main Block:** Do NOT include an `if __name__ == "__main__":` block or any code outside the scene class definition and necessary imports.
        11. **Output:** Output ONLY the raw Python code for the complete script. Do not include any explanations, introductory text, or markdown formatting like ```python ... ```.
        12. **Standard Objects:** Use standard Manim objects (`Text`, `MathTex`, `Square`, `Circle`, `Polygon`, `Line`, `Arrow`, etc.) and animations (`Create`, `Write`, `FadeIn`, `FadeOut`, `Transform`, `Indicate`, `MoveTo`, `Shift`, etc.) to implement the described animations.

        Guideline #1: Environment Context
        The target environment uses **Manim Community v0.18.0** and **manim-voiceover v0.3.7**.
        Ensure all generated code is compatible with these versions. Pay attention to potential API differences from newer Manim versions, especially regarding Camera animations.

        Guideline #2: Code Structure

        Guideline #8: Camera Animations
        Animate the camera using `self.camera.frame.animate...` wrapped in `self.play()` for zooming and panning. Avoid `self.camera.animate` directly. **Look for specific instructions in the plan's `animations` list like "Zoom in on...", "Pan camera to...", or "Set camera width..." and implement them using `self.play(self.camera.frame.animate.scale(...))`, `self.play(self.camera.frame.animate.move_to(...))`, or `self.play(self.camera.frame.animate.set(width=...))` respectively.** Consult the Manim v0.18.0 documentation/examples if unsure.

        Guideline #9: Voiceover Usage

        Generate the Python code now.
        """
        return prompt

# Example Usage (requires a plan object)
# if __name__ == '__main__':
#     # Load a sample plan (e.g., from a file or the PlanningAgent)
#     try:
#         with open("sample_plan.json", "r") as f:
#             sample_plan = json.load(f)
#         video_agent = VideoAgent()
#         code = video_agent.generate_manim_code(sample_plan)
#         if code:
#              print("\n--- Manim Code Generation Successful ---")
#              # Save the code to a file
#              with open("generated_scene.py", "w") as f:
#                  f.write(code)
#              print("Code saved to generated_scene.py")
#         else:
#              print("\n--- Manim Code Generation Failed ---")
#     except FileNotFoundError:
#         print("Error: sample_plan.json not found. Run PlanningAgent first or create a sample.")
#     except Exception as e:
#          print(f"An error occurred: {e}")
