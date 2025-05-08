# zLLM

**zLLM** is a lightweight inference server written in [Zig](https://ziglang.org/) for managing and running LLMs. It supports downloading models from Hugging Face, converting them to the GGUF format, and executing inference using [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## ✨ Features

- Written entirely in Zig
- Integrated model registry (`registry_manifest.json`)
- Supports model download, GGUF conversion, and inference
- MacOS (Metal) support via llama.cpp
- Currently supports OpenAI's chat completions, enough to use Python's OpenAI client library.

---

## 📦 Requirements

- [Zig](https://ziglang.org/download/) 0.14 or newer
- Python 3 virtual environment with dependencies to run `llama.cpp` conversion scripts
  - Required only for `convert` step

---

## 🛠️ Build Locally

```sh
git clone https://github.com/AbeRodz/zLLM.git
cd zLLM
zig build
```
# Usage

The typical usage flow is:

- Download a model
- Convert it to GGUF
- Run inference



## 📥 Get (Download a model)
Downloads a model from Hugging Face.
```sh
zig build run -- get gemma-3-1b 8
```
8 is the number of threads used for downloading.

### NOTE
Various huggingface models required a token and authentication before downloading them e.g. gemma-3. You need to setup the **HF_TOKEN** env variable.

## 🔁 Convert (to GGUF)
Converts the downloaded model using a Python virtual environment.

```sh
zig build run -- convert gemma-3-1b
```

## 🧠 Run (Inference)
Runs inference on the model:
```sh
zig build run -- run gemma-3-1b
```

## 🌐 Run (API)
Runs inference API:
```sh
zig build run -- api-run
```

## ⚙️ Direct Binary Execution

You only need to run zig build once. You can then use the built binary directly:
```sh
./zig-out/bin/zLLM run gemma-3-1b 2>/dev/null
```
2>/dev/null suppresses standard error output from llama.cpp.


## 🖥️ Supported Platforms

Currently only macOS with Metal is supported.

Platform support is bound by build.zig and the capabilities of llama.cpp.

### 📚 Supported Models

The list of supported models is defined in [registry_manifest.json](src/registry/registry_manifest.json). Example entries include:

- gemma-3-1b
- gpt-2

## OpenAI API Support

Currently supports the chat completions endpoint example cURL:

```sh
curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gemma-3-1b",
    "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that only knows about math and nothing more, if what you are asked about is not about math respond just with i dont know."
            },
            {
                "role": "user",
                "content": "i would like about the history of america"
            }
    ],
    "frequency_penalty": 0.5,
    "max_tokens": 100,
    "presence_penalty": 0.3,
    "stop": ["stop"],
    "stream": false,
    "temperature": 0.8,
    "top_p": 1.0
}
'
```

Not all fields from the chat completions are currently supported or do anything per se, but it's enough to use Python's OpenAI client:

```python
import openai

client = openai.OpenAI(
  base_url="http://localhost:8080/v1/", # local zLLM server
  api_key= "apiKey" # dummy apiKey
)

completion = client.chat.completions.create(
    stream=False,
    model="gemma-3-1b", # model saved on registry
    messages=  [
        {
            'role': 'user',
            'content': {
                "text": "I'm doing good"
            },
        }
    ],
    max_tokens=256,
    temperature=0.8,
)

print(completion.choices[0].content)
# i dont know -> expected response given the example above.
```