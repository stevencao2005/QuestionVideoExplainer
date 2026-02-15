import subprocess
import os
import sys
import tempfile # Use tempfile for unique log filenames
import time

def run_manim_script(script_path: str, scene_name: str, output_dir: str, quality: str = "l", attempt: int = 1) -> tuple[bool, str, str]:
    """
    Runs a Manim script using subprocess, redirecting stdout/stderr to files,
    and then reads the content back.

    Args:
        script_path: Path to the Manim Python script.
        scene_name: The name of the Scene class within the script.
        output_dir: Directory to store Manim media output.
        quality: Manim render quality flag ('l', 'm', 'h', 'k'). Defaults to 'l'.
        attempt: The current attempt number (used for log filenames).

    Returns:
        A tuple: (success: bool, full_stdout: str, full_stderr: str)
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True) # Ensure log dir within media dir exists

    # Create unique temporary file paths for stdout and stderr logs
    # We'll use a base name incorporating script, scene, quality, and attempt
    log_basename = f"{os.path.splitext(os.path.basename(script_path))[0]}_{scene_name}_q{quality}_attempt{attempt}"
    stdout_log_path = os.path.join(log_dir, f"{log_basename}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{log_basename}_stderr.log")

    command = [
        "manim",
        "render",
        f"-q{quality}",
        script_path,
        scene_name,
        "--media_dir", output_dir,
        "--log_dir", log_dir,
        "--format", "mp4"
        # We are redirecting stdout/stderr ourselves, so don't need Manim log file flags?
        # However, --log_dir is useful for other Manim internal logs.
    ]

    print(f"\nExecuting Manim command: {' '.join(command)}")
    print(f"  Redirecting stdout to: {stdout_log_path}")
    print(f"  Redirecting stderr to: {stderr_log_path}\n---")

    process = None
    stdout_content = ""
    stderr_content = ""

    try:
        # Open files to capture stdout and stderr
        with open(stdout_log_path, 'w', encoding='utf-8') as f_stdout, \
             open(stderr_log_path, 'w', encoding='utf-8') as f_stderr:
            
            process = subprocess.Popen(
                command,
                stdout=f_stdout,
                stderr=f_stderr,
                text=True, # Still useful for process interaction, though less critical now
                encoding='utf-8',
                errors='replace'
            )

            # Wait for the process to complete
            process.wait()

        returncode = process.returncode
        success = returncode == 0

        # Read the captured output from files
        try:
            with open(stdout_log_path, 'r', encoding='utf-8') as f:
                stdout_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read stdout log file {stdout_log_path}: {e}")
            stdout_content = "<Error reading stdout log>"
            
        try:
            with open(stderr_log_path, 'r', encoding='utf-8') as f:
                stderr_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read stderr log file {stderr_log_path}: {e}")
            stderr_content = "<Error reading stderr log>"
            if success: # If process claimed success but we couldn't read logs, maybe it wasn't truly successful
                 success = False 

        print("\n--- Manim process finished --- ")

        if success:
            print("Manim execution successful (Exit Code 0).")
        else:
            print(f"Manim execution failed (Exit Code {returncode}).")
            # Error details should be in the captured stderr_content

        return success, stdout_content, stderr_content

    except FileNotFoundError:
        print("Error: 'manim' command not found. Is Manim installed and in your PATH?")
        stderr_content = "Manim command not found." # Add error message to stderr
        # No process to kill if FileNotFoundError happens before Popen
        return False, stdout_content, stderr_content
    except Exception as e:
        error_msg = f"An unexpected error occurred while running Manim: {e}"
        print(error_msg)
        if process and process.poll() is None:
            try:
                process.kill()
                process.communicate() # Clean up pipes if any were opened before error
            except Exception as kill_e:
                 print(f"Error trying to kill process after exception: {kill_e}")
        stderr_content += "\n" + error_msg # Append error to stderr
        return False, stdout_content, stderr_content
    # No finally block needed as 'with open' handles file closing
