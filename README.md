# Question Video Explainer

**Turn any question into an animated educational video with AI.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

https://github.com/user-attachments/assets/4e7cca6d-c042-4730-b623-9aa47dcc9cfa

> *"Explain how K-means clustering works"* → 1-minute animated video with narration

---

## Why

I learn best from textbooks, but sometimes a quick visual explanation helps. Making those videos manually takes forever. So I wanted to see: can AI agents do it end-to-end?

This was also a learning project to get hands-on with multi-agent pipelines — planning, generation, self-correction, evaluation — all working together.

---

## What This Does

You give it a question. It gives you back a fully animated explainer video with voiceover.

Under the hood, a pipeline of AI agents:
1. **Plans** the explanation structure
2. **Generates** Manim animation code
3. **Self-corrects** any errors automatically
4. **Narrates** with text-to-speech
5. **Evaluates** the output quality

No manual coding. No video editing. Just question in, video out.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/stevencao2005/QuestionVideoExplainer.git
cd QuestionVideoExplainer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Generate a video
python main.py --question "Explain how binary search works"
```

Output lands in `output/media/videos/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Question                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Planning Agent                            │
│         Breaks down the topic into teachable segments           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Video Agent                              │
│            Generates Manim code + voiceover script              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Linter + Fixing Agent                        │
│     Catches syntax errors, runtime failures, and fixes them     │
│                    (up to 5 retry attempts)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Manim Renderer                             │
│              Produces MP4 video with TTS narration              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evaluate Agent                             │
│         Scores clarity, accuracy, and alignment to question     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Final Video                              │
│                    + Evaluation Report                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Examples

| Topic | Output |
|-------|--------|
| K-Means Clustering | [`examples/kmeans/`](examples/kmeans/) |
| Pythagorean Theorem | [`examples/pythagorean_theorem/`](examples/pythagorean_theorem/) |

---

## Usage

**Generate a new video:**
```bash
python main.py --question "Explain gradient descent intuitively"
```

**Evaluate existing output (skip generation):**
```bash
python main.py --question "Explain gradient descent" --evaluate-only
```

---

## Configuration

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Works with any OpenAI-compatible API endpoint.

---

## Requirements

- Python 3.10+
- [Manim dependencies](https://docs.manim.community/en/stable/installation.html) (LaTeX, ffmpeg, cairo, pango)
- OpenRouter API key (or any OpenAI-compatible endpoint)

**Note:** Manim has system-level dependencies. See their [installation guide](https://docs.manim.community/en/stable/installation.html) if you hit issues.

---

## Project Structure

```
agents/
├── planning_agent.py      # Structures the explanation
├── video_agent.py         # Writes Manim code
├── fixing_agent.py        # Auto-repairs broken code
└── evaluate_agent.py      # Scores final output

utils/
└── manim_runner.py        # Executes Manim rendering

main.py                    # Orchestrates the full pipeline
```

---

## Limitations

- **Not real-time** — generation takes several minutes depending on complexity
- Video quality depends heavily on the underlying LLM's understanding of Manim
- Weak spatial reasoning — shapes often get placed on top of each other or overlap incorrectly
- Complex topics may require multiple generation attempts
- TTS voice is robotic (using gTTS) — swap in ElevenLabs or similar for better audio

---

## License

MIT

---

## Acknowledgments

Built with [Manim](https://www.manim.community/), [manim-voiceover](https://github.com/ManimCommunity/manim-voiceover), and [OpenRouter](https://openrouter.ai/).
