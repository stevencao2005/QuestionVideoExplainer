# Main orchestration script

import argparse
import json
import os
import re # Import re for find_srt_path
import time
import subprocess # Add subprocess import if not already present
import sys # Add sys import for sys.executable

from agents.planning_agent import PlanningAgent
from agents.video_agent import VideoAgent
from agents.fixing_agent import FixingAgent
from utils.manim_runner import run_manim_script
from agents.evaluate_agent import EvaluateAgent # Updated import

# Ensure output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

def run_linter(script_path: str) -> tuple[bool, str]:
    """Runs pylint on the script and returns success status and output."""
    print(f"\n--- Running Linter on {script_path} ---")
    try:
        # We might want to disable certain checks if they are too noisy for generated code
        # Example: disable C0301 (line too long), C0114 (missing module docstring)
        # The exit codes are bit-encoded, 0 means no issues.
        # See: https://pylint.readthedocs.io/en/latest/user_guide/run.html#exit-codes
        # We consider it a "pass" if exit code is 0 (no errors/warnings) or just has convention/refactor/warnings (1, 2, 4)
        # We fail on fatal errors (8), errors (16), or usage errors (32)
        fail_on_codes = 8 | 16 | 32 # Bitwise OR for codes 8, 16, 32
        
        command = [
            sys.executable, # Use the same python interpreter
            "-m", "pylint",
            script_path,
            "--disable=C0114,C0115,C0116,W0614,C0301,C0103", # Disable missing docstrings (common in generated code)
            # Add other disables as needed, e.g., C0301 (line length)
            "--exit-zero", # Exit with 0 even if non-fatal issues are found
            f"--fail-on={fail_on_codes}" # Explicitly fail on specific error codes
        ]
        print(f"Executing: {' '.join(command)}")
        
        # Run pylint
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        # Check the actual exit code against our fail conditions
        # Pylint encodes issue types in bits
        linter_passed = (result.returncode & fail_on_codes) == 0
        
        print(f"Pylint finished with exit code: {result.returncode}")
        print("--- Pylint Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Pylint Stderr ---")
            print(result.stderr)
        print("---------------------")
        
        if linter_passed:
             print("Linter check passed (or only found minor issues).")
             return True, result.stdout # Return stdout for context maybe?
        else:
             print("Linter check failed with critical errors.")
             # Combine stdout and stderr for the fixing agent
             full_output = result.stdout + ("\n--- Pylint Stderr ---\n" + result.stderr if result.stderr else "")
             return False, full_output

    except FileNotFoundError:
        print("Error: 'pylint' command not found or python executable issue. Skipping linting.")
        return True, "Linter skipped: pylint not found." # Treat as pass if pylint isn't runnable
    except Exception as e:
        print(f"An unexpected error occurred during linting: {e}")
        return True, f"Linter skipped: Error {e}" # Treat as pass on unexpected error

def main(question: str, evaluate_only: bool = False):
    print(f"Received question: {question}")

    media_output_dir = os.path.join("output", "media")
    scene_name = "GeneratedVideoScene"
    script_name_base = "generated_scene"
    quality_dir = "480p15"
    code_filename = "output/generated_scene.py"

    render_success = False

    if not evaluate_only:
        # --- 1. Planning --- 
        print("\n--- Generating Plan ---")
        planner = PlanningAgent()
        plan = planner.generate_plan(question)
        if not plan: return

        # Save the plan (optional but good for debugging)
        plan_filename = f"output/plan_{int(time.time())}.json"
        try:
            with open(plan_filename, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"Plan saved to {plan_filename}")
        except Exception as e:
            print(f"Warning: Could not save plan to {plan_filename}. Error: {e}")

        # --- 2. Video Code Generation --- 
        print("\n--- Generating Manim Code ---")
        video_agent = VideoAgent()
        manim_code = video_agent.generate_manim_code(plan)
        if not manim_code: return

        # Initial save of the first generated code
        try:
            with open(code_filename, "w") as f:
                f.write(manim_code)
            print(f"Initial Manim code saved to {code_filename}")
        except Exception as e:
            print(f"Error saving initial Manim code to {code_filename}. Error: {e}")
            return
        
        # --- Pre-Render Linting & Fixing --- 
        linter_passed, linter_output = run_linter(code_filename)
        if not linter_passed:
            print("Attempting to fix linter errors...")
            # Use the FixingAgent to fix based on linter output
            fixer = FixingAgent()
            # Read the code that failed linting
            try:
                 with open(code_filename, "r") as f:
                     code_with_lint_errors = f.read()
            except Exception as e:
                 print(f"Error reading code file {code_filename} for linter fix: {e}")
                 print("Skipping linter fix attempt.")
                 # Optionally proceed to render loop anyway or exit?
                 # For now, let's proceed to render loop
            else:
                fixed_code = fixer.fix_manim_code(code_with_lint_errors, f"Pylint Errors:\n{linter_output}")
                if fixed_code:
                    try:
                        with open(code_filename, "w") as f:
                            f.write(fixed_code)
                        print(f"Applied fix based on linter output and saved to {code_filename}")
                        # Rerun linter to see if fix worked? (Optional, adds complexity)
                        # linter_passed, _ = run_linter(code_filename) 
                        # if not linter_passed:
                        #     print("Warning: Code still has linter errors after fix attempt.")
                    except Exception as e:
                        print(f"Error saving linter-fixed code to {code_filename}: {e}")
                        # Proceed to render loop with potentially unfixed code
                else:
                    print("Fixing agent failed to fix linter errors. Proceeding with potentially faulty code.")
        
        # --- 3. Manim Rendering & Fixing Loop ---
        max_fix_attempts = 5 # Limit retry attempts
        fix_attempts = 0
        while fix_attempts < max_fix_attempts:
            print(f"\n--- Attempting to Render Video (Attempt {fix_attempts + 1}/{max_fix_attempts}) ---")
            try:
                with open(code_filename, "r") as f: current_code_to_render = f.read()
            except Exception as e: print(f"Error reading code: {e}"); break
            
            render_success, stdout, stderr = run_manim_script(
                script_path=code_filename, 
                scene_name=scene_name, 
                output_dir=media_output_dir,
                attempt=fix_attempts + 1
            )
            final_stdout = stdout; final_stderr = stderr
            print("--- Manim Stdout ---"); print(stdout)
            print("--- Manim Stderr ---"); print(stderr)
            print("--------------------")
            if render_success: print("\n--- Video Rendering Successful ---"); break
            else:
                print("\n--- Video Rendering Failed ---")
                fix_attempts += 1
                if fix_attempts >= max_fix_attempts: print("Max fixing attempts reached."); break
                print("Attempting to fix the code...")
                fixer = FixingAgent()
                fixed_code = fixer.fix_manim_code(current_code_to_render, stderr)
                if fixed_code:
                    try:
                        with open(code_filename, "w") as f: f.write(fixed_code)
                        print(f"Attempt {fix_attempts}: Applied fix.")
                    except Exception as e: print(f"Error saving fixed code: {e}"); break
                else: print("Fixer failed."); break
    else:
        # Evaluate Only Mode: Assume render was successful and files exist at expected paths
        print("\n--- Evaluate Only Mode ---")
        render_success = True # Set flag to true to trigger evaluation section

    # --- 4. Evaluation (if rendering succeeded OR evaluate_only mode) ---
    if render_success:
        print("\n--- Determining Output File Paths for Evaluation ---")
        
        # Construct expected paths deterministically
        expected_video_dir = os.path.join(media_output_dir, "videos", script_name_base, quality_dir)
        video_path = os.path.join(expected_video_dir, f"{scene_name}.mp4")
        srt_path = os.path.join(expected_video_dir, f"{scene_name}.srt")

        # Verify paths exist
        video_exists = os.path.exists(video_path)
        srt_exists = os.path.exists(srt_path)

        if video_exists and srt_exists:
            print(f"Verified Video Path: {video_path}")
            print(f"Verified SRT Path:   {srt_path}")
            evaluate_agent = EvaluateAgent() # Renamed instance
            evaluation_results = evaluate_agent.evaluate( # Using renamed instance
                question=question, # Use the original question for context
                video_path=video_path,
                srt_path=srt_path
            )
            print("\n--- Evaluation Results ---")
            # Nicer printing of results
            if "overall_score" in evaluation_results and evaluation_results["overall_score"] is not None:
                 for key, value in evaluation_results.items():
                     # Format score if it's a number, otherwise print as is (for potential errors)
                     score_str = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                     print(f"  {key.replace('_', ' ').title()}: {score_str}")
            else:
                 print("Evaluation could not be fully completed.")
                 print(evaluation_results) # Print raw results/error
        else:
            print("Could not verify expected video and/or SRT file paths. Cannot evaluate.")
            if not video_exists: print(f"  Reason: Expected video path not found: {video_path}")
            if not srt_exists: print(f"  Reason: Expected SRT path not found: {srt_path}")

    print("\n--- Process Complete --- ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate a Manim video explanation.")
    parser.add_argument("-q", "--question", type=str, required=True, help="The question to explain.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip generation and rendering, only run evaluation on existing output files.")
    args = parser.parse_args()
    
    main(args.question, evaluate_only=args.evaluate_only) 