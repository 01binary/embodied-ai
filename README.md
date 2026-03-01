# Embodied AI

Demonstrates how to integrate a large language model with robot hardware by using Tool Calling (also known as Function Calling) functionality supported by most recently published models.

## Overview

This tutorial will walk through the following steps:

+ Select an LLM using LM Studio
+ Write a System Prompt (One-Shot)
+ Provide examples (Few-Shot)
+ Deploy to an industrial PC with LLama.cpp
+ Fine-tune with System Prompt and Few-Shot Prompts

## Tool (Function) Calling

Large Language Models (especially the ones meant for following instructions, with "Instruct" in their name) know how emit Python or JSON in response to natural language prompts, if they have been instructed to do so.

This is known as "tool calling" or "function calling", and it's documented in guides for all major models:

+ [LLama Tool Calling](https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/agents/Agents_Tutorial/Tool_Calling_101.ipynb)
+ [Gemma Function Calling](https://ai.google.dev/gemma/docs/capabilities/function-calling)
+ [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)

## Model Selection

LM Studio provides a simple way to experiment with multiple large language models on your computer. You can load any model from [Hugging Face](https://huggingface.co/models) repository. [LM Studio Community](https://huggingface.co/lmstudio-community) models are listed first.

[Llama.cpp](https://github.com/ggml-org/llama.cpp) is used to run LLMs locally under the hood - exactly the same engine you will be using when deploying your solution to production.

Llama.cpp includes a command line utility called [Llama.cpp Server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), which starts a web server with given IP and port, and loads the specified large language model.

To run a model locally with Llama.cpp, its weights must be available in [GGUF format](https://huggingface.co/docs/hub/en/gguf). Various AI frameworks and Python libraries support converting models in [safetensors](https://huggingface.co/docs/safetensors/en/index) (the most common Open-AI compatible format) to GGUF.

## Model Capabilities

Models are marked with badges to indicate supported features:

|Badge|Capability|
|-|-|
|🛠️ Tool Calling|The model can call external tools by emitting structured JSON|
|🧠 Reasoning|The model is optimized for multi-step reasoning and problem solving|
|👁️ Vision|The model can process and reason about images|

## System Prompt

LM Studio makes it easy to tweak the System Prompt while talking to a model. Simply delete the last message from the model, amend the System Prompt, and click re-generate to ask the same question again.

The system prompt is available on the sidebar in Chat view, along with a few other options like Temperature and Context Size.

## Local Server

LM Studio can run a local web server to interact with loaded LLMs via [Open-AI compatible REST API](https://developers.openai.com/api/reference/overview/).

To talk to a model, you simply send it a list of messages and it will "predict" the next message in that conversation.

Each message in a conversation has a `role` property which can be `system`, `assistant`, or `user`. The system prompt is usually the first message in the conversation with role `system`.

Try sending a POST request to `http://localhost:1234/v1/chat/completions` LLM server with this body:

```json
{
  "model": "gemma",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello! Can you introduce yourself?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 200
}
```

Multi-modal LLMs that support Vision allow you include base-64 encoded images directly in the conversation.

```bash
base64 -w 0 your_image.jpg > image.b64
```

```json
{
  "model": "gemma",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that analyzes images."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What do you see in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,PASTE_YOUR_BASE64_STRING_HERE"
          }
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 300
}
```

Newer versions of Llama Server also support file URLs:

```json
{
  "model": "gemma",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that analyzes images."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What do you see in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///home/ubuntu/Desktop/pinout.png"
          }
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 300
}
```

## Few-Shot Prompts

Using a System Prompt alone to define the model's behavior is called [One-Shot Prompting](https://www.ibm.com/think/topics/one-shot-prompting).

Examples of "ideal behavior" can be included in the list of messages when sending it to get the next message. This is called [Few-Shot Prompting](https://www.ibm.com/think/topics/few-shot-prompting).

You simply include messages in the conversation as if they were written by the AI (with role `assistant`) in response to messages from the user (with role `user`).

This is a great tool for reinforcing the model's behavior if the system prompt alone is not enough because it will learn from the examples provided and attempt to behave the same way.

Loading up a model and talking to it immediately is known as [Zero-Shot Prompting](https://www.ibm.com/think/topics/zero-shot-prompting).

Most open-source LLMs are already fine-tuned to follow instructions and perform fine out of the box. You can also fine-tune your own LLM by starting from a pre-trained or already fine-tuned model, which burns your system prompt and few-shot examples directly into the model weights, enabling zero-shot use in production.

## Deployment

To deploy a locally running model, we will build LLama.cpp from source, download the model from Hugging Face using a Hugging Face token, and run a command to start Llama Server.

### Building Llama.cpp

The following will build the minimum functionality using two threads (`-j2`) to make running out of system memory during build less likely.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

cmake -S . -B build \
  -DGGML_NATIVE=ON \
  -DLLAMA_CUBLAS=OFF \
  -DLLAMA_VULKAN=OFF \
  -DLLAMA_METAL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build -j2
```

### Installing Hugging Face CLI

The following will install minimum Python pre-requisites to run Hugging Face CLI which will let you download open-source large language models:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv pipx
pipx ensurepath
source ~/.bashrc
pipx install huggingface_hub
```

Verify the installation was successful:

```bash
huggingface-cli --version
```

Login to Hugging Face:

```bash
huggingface-cli login

```

### Downloading a Model

Create a directory for the models:

```bash
mkdir -p ~/models
```

Download the model:

> In this example, [Gemma 3 4B](https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF) was used. It's small enough to run on a robotics computer, but great at tool calling and even supports vision (you can send it images and ask what's in them).

```bash
mkdir -p ~/models/gemma-3-4b-it-GGUF

huggingface-cli download \
  lmstudio-community/gemma-3-4b-it-GGUF \
  gemma-3-4b-it-Q4_K_M.gguf \
  --local-dir ~/models/gemma-3-4b-it-GGUF \
  --local-dir-use-symlinks False
```

For multi-modal support, download the **multi-modal projector** (file that typically starts with `mmproj-`):

+ Converts CLIP/Vision encoder output
+ Projects it into Gemma’s token embedding space
+ Required for image inputs

Without this file:
→ Your model is text-only, even though it supports vision.

```bash
huggingface-cli download \
  lmstudio-community/gemma-3-4b-it-GGUF \
  mmproj-model-f16.gguf \
  --local-dir ~/models/gemma-3-4b-it-GGUF \
  --local-dir-use-symlinks False
```

### Running a Model

To run a text-only model:

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf \
  -c 4096 \
  -t 8 \
  -ngl 0 \
  --host 0.0.0.0 \
  --port 1234
```

+ `-m` - model path
+ `-c 4096` - context size
+ `-t 8` - CPU cores (threads)
+ `-ngl 0` - CPU only (not gonna lie?)
+ `--host 0.0.0.0` - run locally
+ `--port 1234` - server port

To run a multi-modal model:

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q3_K_L.gguf \
  --mmproj ~/models/gemma-3-4b-it-GGUF/mmproj-model-f16.gguf \
  --host 127.0.0.1 \
  --port 1234 \
  -c 4096 \
  -ngl 0 \
  --no-mmproj-offload
```

Add `--media-path <path for images>` parameter to support image URLs. Anything after `file://` in image URLs will get added to the path specified here to get the full path.

Install PostMan to test the server:

```bash
sudo snap install postman --classic
postman
```

## Fine-Tuning

Fune-tuning "burns-in" the system prompt and any few-shot examples into model weights, so that when the model is loaded it already knows everything.

This lets you use zero-shot prompts in production, allowing more context memory to be used for talking to real users.

Fine-tuning requires:

- Scientific computing notebook
- Dataset with system prompt and any few-shot prompts

### Notebook

A startup called [Unsloth](https://unsloth.ai/) has developed Hugging Face Transformers extensions that make fine-tuning already trained models much faster.

More importantly, they offer [Python notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) with code already in them that will fine-tune a model of your choice with your personal dataset.

Fine-tuning requires a GPU, and unless you have a gaming rig it's typically done in the cloud using [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).

Here's an [example of a notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) that fine-tunes Gemma 3 4B used in this tutorial.

### Dataset

The most common format for datasets is [JSONL](https://jsonlines.org/). It's a mash-up of CSV and JSON, with rows on separate lines and each row containing a JSON object.

The JSON typically confirms to [ShareGPT](https://sharegpt4o.github.io/) or [Hugging Face Generic Format](https://huggingface.co/docs/trl/main/en/dataset_formats). This looks almost exactly like earlier PostMan examples:

```
{
    messages: [
        {"role": "system", "content": "This is the system prompt"},
        {"role": "user", "content": "This is an example question"},
        {"role": "assistant", "content": "This is an example response"}
    ]
}
```
