import os
import re
import cv2 # OpenCV for video processing
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import base64
import time # Potentially for rate limiting API calls
import json # Added for JSON parsing

# Load environment variables from .env file
load_dotenv()

class EvaluateAgent:
    def __init__(self):
        # Load environment variables for OpenRouter/OpenAI
        self.api_key = os.getenv("OPENROUTER_API_KEY") # Using OpenRouter key as default
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.site_url = os.getenv("SITE_URL", "http://localhost:3000") # Added for headers
        self.project_title = os.getenv("PROJECT_TITLE", "QuestionVideoExplainer") # Added for headers

        if not self.api_key:
            raise ValueError("API key (OPENROUTER_API_KEY) not found in .env file")

        # Prepare headers for OpenRouter (especially needed for free models)
        self.extra_headers = None
        if "openrouter.ai" in self.base_url:
             self.extra_headers = {
                "HTTP-Referer": self.site_url, 
                "X-Title": self.project_title
            }
        # --- Debug: Print headers --- REMOVED
        # print(f"[Debug EvaluateAgent] Initialized with base_url: {self.base_url}")
        # print(f"[Debug EvaluateAgent] Using headers: {self.extra_headers}")
        # --- End Debug ---

        # Initialize OpenAI client (adjust if using direct OpenAI API)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        # Store model names for easier changes
        self.text_model = "openai/gpt-4o" # Recommended for text/code tasks
        # self.vision_model = "openai/gpt-4o" # Recommended for vision tasks
        self.vision_model = "openai/gpt-4o" # Use GPT-4o for Vision to avoid free Gemini limits
        # Placeholder for video model - adjust based on availability/API
        # Check OpenAI documentation for current video-capable models accessible via API
        # self.video_model = "openai/gpt-4o" # Assuming GPT-4o handles video, might need adjustment
        self.video_model = "openai/gpt-4o" # Use GPT-4o for fallback frame consistency

    def _parse_srt(self, srt_path: str) -> tuple[str, list[dict]]:
        """Parses an SRT file into a full transcript string and a list of subtitle entries with timings."""
        if not os.path.exists(srt_path):
            print(f"Error: SRT file not found at {srt_path}")
            return "", []
            
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading SRT file {srt_path}: {e}")
            return "", []

        # Basic SRT parsing (adjust regex if format varies)
        subtitle_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.DOTALL)
        
        subtitles = []
        full_transcript = []
        
        for match in subtitle_pattern.finditer(content):
            index = int(match.group(1))
            start_time_str = match.group(2)
            end_time_str = match.group(3)
            text = match.group(4).strip().replace('\n', ' ') # Join multi-line subtitles
            
            # Convert time strings to seconds (simple helper)
            def time_str_to_seconds(time_str):
                h, m, s_ms = time_str.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

            subtitles.append({
                'index': index,
                'start': time_str_to_seconds(start_time_str),
                'end': time_str_to_seconds(end_time_str),
                'text': text
            })
            full_transcript.append(text)
            
        return " ".join(full_transcript), subtitles

    def _extract_key_frames(self, video_path: str, subtitles: list[dict], num_frames_per_subtitle=1) -> list[tuple[float, np.ndarray]]:
        """Extracts key frames from the video, using subtitle timings."""
        key_frames = []
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: Could not get FPS from video. Frame extraction might be inaccurate.")
            fps = 30 # Assume default if unknown

        for sub in subtitles:
            start_time = sub['start']
            end_time = sub['end']
            duration = end_time - start_time

            # Extract frame(s) within the subtitle duration
            for i in range(num_frames_per_subtitle):
                 # Aim for middle of the subtitle segment or evenly spaced
                time_offset = duration * (i + 1) / (num_frames_per_subtitle + 1)
                target_time = start_time + time_offset
                frame_index = int(target_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    # Store frame with its timestamp
                    key_frames.append((target_time, frame))
                else:
                     print(f"Warning: Could not read frame at time {target_time:.2f}s (index {frame_index})")

        cap.release()
        print(f"Extracted {len(key_frames)} key frames.")
        return key_frames
        
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encodes a single OpenCV frame to a base64 string for API use."""
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Warning: Failed to encode frame to JPEG.")
            return ""
        return base64.b64encode(buffer).decode('utf-8')

    # --- Evaluation Dimension Methods (Placeholders) ---

    def _evaluate_accuracy_depth(self, question: str, transcript: str) -> float:
        """Evaluates factual accuracy and depth of explanation using an LLM."""
        print("Evaluating Accuracy & Depth...")
        
        prompt = f"""
        Analyze the following transcript, which is intended to answer the question: "{question}"

        Transcript:
        {transcript}

        Evaluation Criteria (relative to the request for a SIMPLE explanation):
        1.  **Accuracy:** Is the core explanation factually correct?
        2.  **Sufficient Detail:** Does it provide enough detail for a simple, introductory understanding?

        Based on these criteria, determine a single score between 0.0 and 1.0. A score of 1.0 means it's accurate and sufficiently detailed for a simple explanation. A score of 0.0 means it's fundamentally inaccurate or completely lacks necessary detail.

        FINAL INSTRUCTION: Call the 'submit_evaluation_score' tool with the calculated score.
        """
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": "submit_evaluation_score",
                "description": "Submit the evaluation score based on the analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "The evaluation score between 0.0 and 1.0.",
                        }
                    },
                    "required": ["score"],
                },
            },
        }

        # --- Add Debug Print Here --- REMOVED: Transcript print (redundant for now)
        # print(f"--- Debug: Transcript for Accuracy ---\n{transcript}\n--------------------------------------")
        # --- End Debug Print ---

        response = None # Define response outside try for logging
        # response_content = "" # Initialize for error logging -- Not needed for tool call args
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[tool_schema], # Provide tool schema
                tool_choice={"type": "function", "function": {"name": "submit_evaluation_score"}}, # Force tool use
                temperature=0, # Ensure deterministic output
                extra_headers=self.extra_headers, # Pass headers
                # REMOVED: response_format={'type': 'json_object'}
            )
            
            # --- Add Debug Print Here --- REMOVED: Raw response print (less relevant now)
            # print(f"--- Debug: Raw Accuracy Response ---\n{response}------------------------------------------")
            # --- End Debug Print ---
            
            # Extract arguments from tool call
            tool_call = response.choices[0].message.tool_calls[0] # Assuming one tool call
            if tool_call.function.name != "submit_evaluation_score":
                raise ValueError(f"Expected tool 'submit_evaluation_score' but got '{tool_call.function.name}'")

            arguments_str = tool_call.function.arguments
            # --- Debug Print Removed ---
            # print(f"--- Debug: Tool Call Arguments (Accuracy) ---\n{arguments_str}\n-------------------------------------------------")
            arguments = json.loads(arguments_str)
            
            # Extract score, convert to float, handle potential errors
            score_value = arguments.get("score")
            if score_value is None:
                raise ValueError("Tool call arguments missing 'score' key")
            
            score = float(score_value)

            # Clamp the score to the valid range [0.0, 1.0]
            score = max(0.0, min(1.0, score))
            print(f"  Accuracy & Depth Score: {score:.3f}")
            return score
        except (IndexError, AttributeError) as e:
            print(f"Error accessing tool call data during Accuracy & Depth evaluation: {e}")
            print(f"  Raw response object: {response}")
            return 0.0
        except json.JSONDecodeError as e:
             print(f"Error decoding JSON from tool arguments during Accuracy & Depth evaluation: {e}")
             print(f"  Raw arguments string: {arguments_str}")
             return 0.0 # Return default score on JSON error
        except (ValueError, TypeError) as e:
            print(f"Error processing score from tool arguments during Accuracy & Depth evaluation: {e}")
            if 'arguments_str' in locals(): print(f"  Raw arguments string: {arguments_str}")
            return 0.0 # Return default score on other value errors
        except Exception as e:
            print(f"Unexpected error during Accuracy & Depth evaluation: {e}")
            # Enhanced error logging
            if response:
                 print(f"  Raw response object: {response}")
            return 0.0 # Return a default low score on error

    def _evaluate_logical_flow(self, question: str, transcript: str) -> float:
        """Evaluates the logical flow and coherence of the transcript."""
        print("Evaluating Logical Flow...")
        
        prompt = f"""
        Analyze the following transcript, which is intended to answer the question: "{question}"

        Transcript:
        """
        {transcript}
        """

        Evaluation Criteria:
        1.  **Logical Flow:** Does the explanation follow a clear, logical sequence? Do ideas build upon each other effectively?
        2.  **Coherence:** Is the explanation easy to follow? Are transitions between concepts smooth?
        3.  **Structure:** Is the overall structure well-organized for explaining the answer to the specific question?

        Based on these criteria, determine a single score between 0.0 and 1.0, where 0.0 represents a completely illogical or incoherent explanation, and 1.0 represents a perfectly structured and coherent explanation for the question asked.

        FINAL INSTRUCTION: Call the 'submit_evaluation_score' tool with the calculated score.
        """
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": "submit_evaluation_score",
                "description": "Submit the evaluation score based on the analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "The evaluation score between 0.0 and 1.0.",
                        }
                    },
                    "required": ["score"],
                },
            },
        }
        
        # --- Add Debug Print Here --- REMOVED
        # print(f"--- Debug: Transcript for Logical Flow ---\n{transcript}\n----------------------------------------")
        # --- End Debug Print ---

        response = None
        # response_content = "" # Not needed
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[tool_schema], # Provide tool schema
                tool_choice={"type": "function", "function": {"name": "submit_evaluation_score"}}, # Force tool use
                temperature=0, # Ensure deterministic output
                extra_headers=self.extra_headers, # Pass headers
                # REMOVED: response_format={'type': 'json_object'}
            )
            
            # --- Add Debug Print Here --- REMOVED
            # print(f"--- Debug: Raw Logical Flow Response ---\n{response}------------------------------------------")
            # --- End Debug Print ---
            
            # Extract arguments from tool call
            tool_call = response.choices[0].message.tool_calls[0] # Assuming one tool call
            if tool_call.function.name != "submit_evaluation_score":
                raise ValueError(f"Expected tool 'submit_evaluation_score' but got '{tool_call.function.name}'")
            
            arguments_str = tool_call.function.arguments
            # --- Debug Print Removed ---
            # print(f"--- Debug: Tool Call Arguments (Logical Flow) ---\n{arguments_str}\n-------------------------------------------------")
            arguments = json.loads(arguments_str)
            
            # Extract score, convert to float, handle potential errors
            score_value = arguments.get("score")
            if score_value is None:
                raise ValueError("Tool call arguments missing 'score' key")
            
            score = float(score_value)

            # Clamp the score to the valid range [0.0, 1.0]
            score = max(0.0, min(1.0, score))
            print(f"  Logical Flow Score: {score:.3f}")
            return score
        except (IndexError, AttributeError) as e:
            print(f"Error accessing tool call data during Logical Flow evaluation: {e}")
            print(f"  Raw response object: {response}")
            return 0.0
        except json.JSONDecodeError as e:
             print(f"Error decoding JSON from tool arguments during Logical Flow evaluation: {e}")
             print(f"  Raw arguments string: {arguments_str}")
             return 0.0 # Return default score on JSON error
        except (ValueError, TypeError) as e:
            print(f"Error processing score from tool arguments during Logical Flow evaluation: {e}")
            if 'arguments_str' in locals(): print(f"  Raw arguments string: {arguments_str}")
            return 0.0 # Return default score on other value errors
        except Exception as e:
            print(f"Unexpected error during Logical Flow evaluation: {e}")
            if response:
                 print(f"  Raw response object: {response}")
            return 0.0 # Return a default low score on error

    def _evaluate_visual_relevance(self, key_frames: list[tuple[float, np.ndarray]], subtitles: list[dict]) -> float:
        """Evaluates alignment between visuals (key frames) and narration segments."""
        print(f"Evaluating Visual Relevance for {len(key_frames)} frames...")
        scores = []
        if not key_frames: return 0.0

        # Use a fraction of frames to avoid excessive API calls/cost during testing
        # Set sample_rate=1.0 to use all extracted frames
        sample_rate = 0.5 # Evaluate 50% of the key frames
        sampled_frames = key_frames[::int(1/sample_rate)] if sample_rate > 0 else []
        print(f"  (Using {len(sampled_frames)} frames for evaluation based on sample rate {sample_rate})" if sample_rate < 1.0 else "")

        if not sampled_frames: return 0.0

        for i, (timestamp, frame) in enumerate(sampled_frames):
            # Find corresponding subtitle text
            relevant_text = "(No specific subtitle text found for this frame's timestamp)"
            for sub in subtitles:
                # Check if frame timestamp falls within subtitle duration
                if sub['start'] <= timestamp <= sub['end']:
                    relevant_text = sub['text']
                    break # Use the first matching subtitle

            base64_frame = self._encode_frame(frame)
            if not base64_frame:
                print(f"  Skipping frame {i+1} due to encoding error.")
                continue
            
            print(f"  Evaluating frame {i+1}/{len(sampled_frames)} at {timestamp:.2f}s...")

            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator assessing the visual relevance of video frames to their corresponding narration. Output only a numerical score between 0.0 and 1.0."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Consider the narration segment: \"{relevant_text}\"\n\nHow well do the visual elements in the provided image frame illustrate, support, or visually correspond to this narration? Evaluate the relevance on a scale from 0.0 (completely irrelevant or contradictory) to 1.0 (perfectly illustrative and supportive). Output ONLY the numerical score."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        }
                    ]
                }
            ]

            # --- Add Debugging for Prompt --- REMOVED
            # print(f"\n--- Debug: Visual Relevance Prompt (Frame {i+1}) ---")
            # print(prompt_messages[1]['content']) # Print the user content
            # print("------------------------------------------")
            # --- End Debugging ---

            response = None
            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=prompt_messages,
                    temperature=0,
                    max_tokens=10, # Expecting just a score
                    extra_headers=self.extra_headers # Pass headers
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text)
                score = max(0.0, min(1.0, score)) # Clamp score
                scores.append(score)
                print(f"    Frame {i+1} Score: {score:.3f}")

            except Exception as e:
                print(f"Error during Visual Relevance evaluation for frame {i+1}: {e}")
                if response and response.choices:
                    choice = response.choices[0]
                    print(f"  LLM Response Content: '{choice.message.content}'")
                    print(f"  LLM Finish Reason: {choice.finish_reason}")
                else:
                    print(f"  LLM Response object or choices missing. Response: {response}")
                scores.append(0.0) # Assign low score on error for this frame
            
            # Basic rate limiting to avoid overwhelming the API
            time.sleep(2) # Increased sleep time for potentially heavier vision calls

        final_score = np.mean(scores) if scores else 0.0
        print(f"  Visual Relevance Average Score: {final_score:.3f}")
        return final_score

    def _evaluate_element_layout(self, key_frames: list[tuple[float, np.ndarray]]) -> float:
        """Evaluates the layout, positioning, and clarity of visual elements."""
        print(f"Evaluating Element Layout for {len(key_frames)} frames...")
        scores = []
        if not key_frames: return 0.0

        # Use a fraction of frames to avoid excessive API calls/cost during testing
        # Set sample_rate=1.0 to use all extracted frames
        sample_rate = 0.5 # Evaluate 50% of the key frames
        sampled_frames = key_frames[::int(1/sample_rate)] if sample_rate > 0 else []
        print(f"  (Using {len(sampled_frames)} frames for evaluation based on sample rate {sample_rate})" if sample_rate < 1.0 else "")

        if not sampled_frames: return 0.0

        for i, (timestamp, frame) in enumerate(sampled_frames):
            base64_frame = self._encode_frame(frame)
            if not base64_frame:
                print(f"  Skipping frame {i+1} due to encoding error.")
                continue
            
            print(f"  Evaluating frame {i+1}/{len(sampled_frames)} at {timestamp:.2f}s...")

            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator assessing the visual layout quality of video frames. Output only a numerical score between 0.0 and 1.0."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Evaluate the element layout in the provided image frame. Consider the following:\n- Positioning: Are elements placed thoughtfully?\n- Sizing: Are elements appropriately sized relative to each other and the frame?\n- Overlap: Is there distracting or confusing overlap between elements?\n- Clarity: Is the overall presentation clear and uncluttered?\n\nBased on these criteria, provide a single score between 0.0 (very poor layout, cluttered, confusing) and 1.0 (excellent layout, clear, well-organized). Output ONLY the numerical score."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        }
                    ]
                }
            ]

            response = None
            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=prompt_messages,
                    temperature=0,
                    max_tokens=10, # Expecting just a score
                    extra_headers=self.extra_headers # Pass headers
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text)
                score = max(0.0, min(1.0, score)) # Clamp score
                scores.append(score)
                print(f"    Frame {i+1} Score: {score:.3f}")

            except Exception as e:
                print(f"Error during Element Layout evaluation for frame {i+1}: {e}")
                if response and response.choices:
                    choice = response.choices[0]
                    print(f"  LLM Response Content: '{choice.message.content}'")
                    print(f"  LLM Finish Reason: {choice.finish_reason}")
                else:
                    print(f"  LLM Response object or choices missing. Response: {response}")
                scores.append(0.0) # Assign low score on error for this frame
            
            # Basic rate limiting
            time.sleep(2)

        final_score = np.mean(scores) if scores else 0.0
        print(f"  Element Layout Average Score: {final_score:.3f}")
        return final_score

    def _evaluate_visual_consistency(self, video_path: str, key_frames: list[tuple[float, np.ndarray]]) -> float:
        """Evaluates visual style consistency across frames using a vision model (fallback)."""
        print("Evaluating Visual Consistency (Frame-based Fallback)...")
        # This method does NOT evaluate motion smoothness effectively.
        
        if not key_frames: return 0.0

        # Sample frames across the video duration for consistency check
        # Use a small number of frames spread out to avoid excessive token usage
        num_consistency_frames = 15 # Increased number of frames
        if len(key_frames) < num_consistency_frames:
            sampled_indices = np.arange(len(key_frames))
        else:
            sampled_indices = np.linspace(0, len(key_frames) - 1, num_consistency_frames, dtype=int)
        
        sampled_frames_data = []
        for idx in sampled_indices:
            timestamp, frame = key_frames[idx]
            base64_frame = self._encode_frame(frame)
            if base64_frame:
                sampled_frames_data.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
                })
            else:
                 print(f"  Warning: Could not encode frame at index {idx} for consistency check.")

        if len(sampled_frames_data) < 2: # Need at least 2 frames to compare consistency
            print("  Warning: Not enough valid frames sampled to evaluate consistency.")
            return 0.5 # Return neutral score if too few frames

        print(f"  (Using {len(sampled_frames_data)} frames for evaluation)")

        prompt_text = (
            "Evaluate the visual style consistency ACROSS THE SEQUENCE of the provided image frames from a video. Consider elements like:\n"
            "- Color palette: Are colors used consistently between frames?\n"
            "- Object appearance: Do recurring objects maintain a consistent style (shape, line weight, texture) across frames?\n"
            "- Text style: Are fonts, sizes, and colors for text elements consistent where expected?\n"
            "- Overall aesthetic: Does the video maintain a coherent visual style throughout this sequence? Are there abrupt or jarring changes in style between frames?\n\n"
            "Based on these criteria, provide a single score between 0.0 (very inconsistent style, jarring changes) and 1.0 (very consistent style across the sequence). "
            "Note: This assesses static style consistency across frames, not complex animation smoothness. Output ONLY the numerical score."
        )

        prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing the visual style consistency of video frames. Output only a numerical score between 0.0 and 1.0."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ] + sampled_frames_data # Append the list of image data dictionaries
            }
        ]

        response = None
        try:
            # Using self.video_model which we set to the GPT-4o model
            response = self.client.chat.completions.create(
                model=self.video_model,
                messages=prompt_messages,
                temperature=0,
                max_tokens=10, # Expecting just a score
                extra_headers=self.extra_headers # Pass headers
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            score = max(0.0, min(1.0, score)) # Clamp score
            print(f"  Visual Consistency Score: {score:.3f}")
            return score

        except Exception as e:
            print(f"Error during Visual Consistency evaluation: {e}")
            if response and response.choices:
                choice = response.choices[0]
                print(f"  LLM Response Content: '{choice.message.content}'")
                print(f"  LLM Finish Reason: {choice.finish_reason}")
            else:
                print(f"  LLM Response object or choices missing. Response: {response}")
            return 0.0 # Assign low score on error

    # --- Main Evaluation Method ---

    def evaluate(self, question: str, video_path: str, srt_path: str) -> dict:
        """
        Performs a comprehensive evaluation of the generated video based on the paper's metrics.

        Args:
            question: The original question the video aims to answer.
            video_path: Path to the generated MP4 video file.
            srt_path: Path to the generated SRT subtitle file.

        Returns:
            A dictionary containing scores for each dimension and the overall geometric mean.
        """
        print("\n--- Starting Evaluation --- ")
        if not all([os.path.exists(video_path), os.path.exists(srt_path)]):
            print("Error: Video or SRT file not found.")
            return {"error": "Input file(s) missing."}

        # 1. Preprocessing
        transcript, subtitles = self._parse_srt(srt_path)
        if not transcript or not subtitles:
            print("Error: Failed to parse SRT file.")
            return {"error": "SRT parsing failed."}
            
        key_frames = self._extract_key_frames(video_path, subtitles)
        # We might need more frames for layout/consistency if not doing video analysis
        # key_frames_layout = self._extract_key_frames(video_path, subtitles, num_frames_per_subtitle=2) # Example

        # 2. Evaluate Dimensions
        scores = {}
        scores['accuracy_depth'] = self._evaluate_accuracy_depth(question, transcript)
        scores['logical_flow'] = self._evaluate_logical_flow(question, transcript)
        scores['visual_relevance'] = self._evaluate_visual_relevance(key_frames, subtitles)
        scores['element_layout'] = self._evaluate_element_layout(key_frames) # Use same frames or different sample?
        scores['visual_consistency'] = self._evaluate_visual_consistency(video_path, key_frames)

        # Filter out any potential None/Error scores before calculating geometric mean
        valid_scores = [s for s in scores.values() if isinstance(s, (int, float)) and 0.0 <= s <= 1.0]

        # 3. Aggregate Score (Geometric Mean)
        if not valid_scores or len(valid_scores) != 5:
             print(f"Warning: Could not calculate overall score. Only got {len(valid_scores)} valid dimension scores.")
             overall_score = None
        else:
            # Avoid issues with log(0) if a score is exactly 0. Add small epsilon or handle differently?
            # Let's cap scores slightly above 0 for geo mean calculation.
            epsilon = 1e-9
            adjusted_scores = [max(s, epsilon) for s in valid_scores]
            overall_score = np.exp(np.mean(np.log(adjusted_scores)))
            scores['overall_score'] = overall_score
            print(f"--- Evaluation Complete ---")
            print(f"  Accuracy & Depth: {scores['accuracy_depth']:.3f}")
            print(f"  Logical Flow:     {scores['logical_flow']:.3f}")
            print(f"  Visual Relevance: {scores['visual_relevance']:.3f}")
            print(f"  Element Layout:   {scores['element_layout']:.3f}")
            print(f"  Visual Consist.:  {scores['visual_consistency']:.3f}")
            print(f"---------------------------")
            print(f"  Overall Score:    {overall_score:.3f}")
            print(f"---------------------------")

        return scores 