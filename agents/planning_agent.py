import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PlanningAgent:
    def __init__(self):
        # Load environment variables for OpenRouter
        self.api_key = os.getenv("OPENROUTER_API_KEY") 
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") # Default to OpenRouter URL

        # --- Debugging Print Statements ---
        # print(f"Debug: Loaded OPENROUTER_API_KEY: {self.api_key[:5]}...{self.api_key[-4:] if self.api_key else 'None'}")
        # print(f"Debug: Loaded OPENAI_BASE_URL: {self.base_url}")
        # --- End Debugging --- 
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        # Initialize OpenAI client pointing to OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # Define the expected JSON structure (can be refined)
        self.plan_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Catchy title for the video."},
                "scenes": {
                    "type": "array",
                    "description": "List of scenes in the video.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "integer", "description": "Unique identifier for the scene, starting from 1."},
                            "narration": {"type": "string", "description": "The voiceover script for this scene."},
                            "animations": {
                                "type": "array",
                                "description": "List of animations described in natural language.",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["scene_id", "narration", "animations"]
                    }
                }
            },
            "required": ["title", "scenes"]
        }


    def generate_plan(self, question: str) -> dict:
        """
        Generates a video plan based on the user's question using OpenRouter.

        Args:
            question: The user's question.

        Returns:
            A dictionary representing the structured video plan.
            Returns None if plan generation fails.
        """
        prompt = self._create_prompt(question)
        
        # Prepare headers for OpenRouter
        extra_headers = None
        # Check if using the standard OpenRouter URL
        if self.base_url == 'https://openrouter.ai/api/v1':
             extra_headers = {
                "HTTP-Referer": "http://localhost:3000", # Replace with your actual site URL 
                "X-Title": "QuestionVideoExplainer"      # Replace with your actual site name
            }

        try:
            response = self.client.chat.completions.create(
                model="openai/o3-mini-high",
                messages=[
                    {"role": "system", "content": "You are an expert in educational video production and instructional design, specializing in creating clear visual explanations using Manim. You will generate a JSON plan for an explanatory video."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object", "schema": self.plan_schema}, # Note: Check if the chosen model fully supports strict JSON mode
                temperature=0.7, 
                extra_headers=extra_headers # Pass headers here
            )
            
            plan_json = response.choices[0].message.content
            plan = json.loads(plan_json)
            # Use triple quotes for multi-line f-string
            print(f"""Generated Plan:
            {json.dumps(plan, indent=2)}""") # Log the generated plan
            return plan

        except Exception as e:
            print(f"Error generating video plan: {e}")
            # Consider more robust error handling/logging
            return None

    def _create_prompt(self, question: str) -> str:
        """Helper function to create the prompt for the LLM."""
        # Use f-string for cleaner formatting
        prompt = f"""
        Design a structured video plan (in JSON format) for a short (3-7 scenes) Manim animation explaining the following topic/question clearly and concisely:
        '{question}'

        **Output Requirements:**
        - The output MUST be a single JSON object conforming exactly to the schema below.
        - Do NOT include any text outside the JSON object.

        **JSON Schema:**
        ```json
        {json.dumps(self.plan_schema, indent=2)}
        ```

        **Instructional Design Guidelines:**
        1.  **Progressive Structure:** Design scenes to build logically, starting with foundational concepts and progressing to more complex ideas. Ensure a smooth narrative flow.
        2.  **Clarity:** Focus on clear visual storytelling. Avoid unnecessary clutter.
        3.  **Scene Content:**
            - `title`: Create a concise and engaging video title.
            - `narration`: Write a clear, spoken script for each scene.
            - `animations`: Describe the visual elements and animations for each scene in natural language. **Include VERY specific hints about spatial layout and relative positioning** (e.g., "Draw square in center", "Place text 0.5 units BELOW the square, aligned LEFT", "Arrange formula and diagram vertically with 1.0 buff between them", "Ensure text doesn't overlap the triangle"). Also, suggest dynamic camera movements where appropriate for emphasis (e.g., "Zoom in on the formula", "Pan camera to follow the moving object") and describe smooth transitions between scenes or major elements (e.g., "Fade out the previous diagram before showing the next"). These explicit layout and dynamic hints are CRITICAL for guiding the code generation later and preventing overlaps.
        4.  **Educational Focus:** Explain the topic thoroughly. Do not include promotional content or quizzes.
        5.  **Conciseness:** Keep the overall plan suitable for a relatively short video.

        Generate the JSON plan now.
        """
        return prompt

# Example Usage (for testing)
if __name__ == '__main__':
    planner = PlanningAgent()
    test_question = "Explain the Pythagorean theorem."
    plan = planner.generate_plan(test_question)
    
    if plan:
        print("\n--- Plan Generation Successful ---")
        # Further processing or saving the plan could happen here
    else:
        print("\n--- Plan Generation Failed ---")
