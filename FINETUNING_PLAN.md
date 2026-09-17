# Fine-Tuning Plan: Sharpening the Local Qwen Model on Data-Science Skills

Status: plan for later use. Written 2026-09-17. Not yet started.

## Purpose

The agent runs on `qwen3.8:27b-mlx` through Ollama. That model is a strong general model trained across many tasks, but it is not specialised for the work this agent does most: data analysis, data-science and machine-learning decisions, and use of AI APIs. This document records how to sharpen the model on those skills using two related methods, distillation from Claude and a preference-optimization step that plays the role of RLHF. It is a reference for a future effort, not a task in progress.

## The two methods and the order they run in

The two ideas map onto two established methods, and they run in sequence rather than as alternatives.

Distillation from Claude is supervised fine-tuning on high-quality outputs that Claude produces for data-science tasks. Claude is the teacher and Qwen is the student. This step raises the floor. The model learns the response format, the discipline of calling tools correctly, and the shape of a good analysis.

The preference-optimization step is the part that resembles RLHF. The practical modern form is Direct Preference Optimization, trained on pairs where one answer is better than the other. When the judge is a model rather than a person, the field calls this reinforcement learning from AI feedback. This step sharpens the model, because it learns to prefer the better of two candidate answers. Full reinforcement learning with a separate reward model and PPO is heavier and is not worth it at this scale. Direct Preference Optimization captures most of the benefit with far less machinery.

The standard pipeline is supervised fine-tuning first to teach the skill, then Direct Preference Optimization to sharpen the preferences.

## Constraints to respect

The Ollama model cannot be fine-tuned directly. The file `qwen3.8:27b-mlx` is a quantized MLX artifact meant for inference. Training needs the base weights in a trainable format, which means the Hugging Face safetensors for that Qwen family or an MLX float conversion. The recipe is to train LoRA or QLoRA adapters on the base weights, then merge, convert, and re-quantize for Ollama. The first task is therefore to confirm which upstream base checkpoint this MLX build came from. If only the quantized build is available, fine-tune the nearest open base of the same family and re-serve it.

Size decides where training runs. At 27.8 billion parameters, full fine-tuning is not possible on the M1 Pro with 32 GB. LoRA through mlx-lm on the 27B model is marginal even in 4-bit on 32 GB of unified memory. A realistic local proof of concept uses a 7B or 14B Qwen. The 27B model belongs on the remote Slurm CUDA node, where QLoRA on a GPU with 24 GB or more is comfortable. The recommended approach is to validate the whole pipeline on a small model locally and then scale the same recipe to the 27B model on the cluster.

## Data

Model quality here is data quality, so most of the effort goes into building good training data.

Build a task taxonomy across the three target skills. Data analysis covers reading distributions, spotting data issues, and deciding what to explore next. Data science and machine learning cover choosing a model from the shape of the data, feature engineering, picking a metric, and reading results. AI-API usage covers Anthropic API calls, tool definitions, and agent patterns. Supporting code in pandas, scikit-learn, polars, and SQL sits underneath all three.

Generate three kinds of examples.

The first kind is supervised fine-tuning pairs. Each pair is a task and a dataset context together with Claude's ideal response, which may be an analysis, a plan, code, or a tool-call sequence. Use synthetic datasets with known structure so the answers can be checked rather than trusted.

The second kind is on-policy corrections. Run the agent loop, capture each state, the tool call it chose, and the output, then have Claude produce the better next step. This distills the decision policy of the agent, not only its prose, and it reuses the critic and the exploration log already built into the agent.

The third kind is preference pairs for Direct Preference Optimization. For each prompt, produce two candidates, either the model's own sample against Claude's or two of the model's own samples, and have Claude rank them. That ranking is the preference signal.

The running agent already produces most of this data for free. The critic labels every step as accept, retry, or replan and writes a one-line finding. A retry or a replan is a rejected action, and the corrected next attempt is the chosen one, so ordinary use yields preference pairs on its own. Approved full traces are supervised fine-tuning material. Instrument the agent to log the context, the tool call, the output, the verdict, and the finding to JSONL, and the result is a collector that grows with every session. This is the data flywheel.

## Evaluation and guarding the model

Hold out a data-science evaluation set and score several things: task success, tool-call validity such as whether the JSON parses and names a real tool, whether the generated code runs, whether the metric is correct, and whether the SQL is correct. Measure before and after each stage so the gain is visible and regressions are caught.

Mix a modest amount of general instruction data into the supervised set so the model specialises on data science without losing broad ability. Keep a small general evaluation alongside the specialised one.

Version the datasets and the adapters. The repository already runs DVC and MLflow in `retrain.yml` for the data-science models, and the same discipline applied to the LLM adapters makes a bad run reversible.

## Stack

Generate data through the Anthropic API with a script that emits JSONL and uses structured outputs. Train with mlx-lm for LoRA and Direct Preference Optimization locally on the small model, then with TRL and PEFT using QLoRA on the Slurm node for the 27B model. Both stacks support supervised fine-tuning and Direct Preference Optimization. Serve the result by merging the adapter and converting to GGUF or MLX for Ollama with a Modelfile, or by serving the fine-tuned model with mlx-lm or vLLM and pointing the agent's model name and host at it. The agent code needs no change beyond the model name.

## Caveat on training data

Training a model on Claude's outputs falls under Anthropic's usage policies. Using it for personal research and an internal agent is a different matter from building and distributing a model that competes with Claude. Be clear about the intended use before generating a large distillation set. This is a consideration to settle up front, not a technical blocker.

## Suggested order of work

Phase 0. Decide three things: the target model size and where it will train, the availability of the base weights for this Qwen family, and the intended use of the distilled data.

Phase 1. Instrument the agent to log traces to JSONL. This is a small change that reuses the critic and the exploration log, and it begins collecting data immediately, before any training.

Phase 2. Build the Claude data generator for supervised pairs and preference pairs across the taxonomy, and fold in the logged traces.

Phase 3. Run supervised fine-tuning with LoRA on the small model locally to validate the pipeline from end to end, then evaluate.

Phase 4. Run Direct Preference Optimization on the Claude-judged pairs, then evaluate.

Phase 5. Convert, serve, swap the agent's model, measure end to end, and iterate the flywheel.

The cheapest first move with the highest leverage is Phase 1, the trace logger, because it turns every session into training data and reuses the reflection machinery already in place.
