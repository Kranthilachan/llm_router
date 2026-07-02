"""
Raw provider -> model-name catalog. This is the "menu" of candidate models
known to the router. It carries no operational metadata by itself — the
registry builder (registry_builder.py) turns each name into a fully-shaped
canonical registry entry (modalities, capabilities, pricing, priors, etc).

Keeping the catalog separate from the registry builder makes it trivial to
add/remove a model without touching scoring logic.
"""

MODELS = {
    "OpenAI": [
        "GPT-5.5", "GPT-5.5-Thinking", "GPT-5.5-Instant", "GPT-5.4",
        "GPT-5.4-Mini", "GPT-5", "GPT-4o", "GPT-4o-Mini", "o4-Mini", "o3"
    ],
    "Anthropic": [
        "Claude-Opus-4.8", "Claude-Opus-4.7", "Claude-Opus-4.6",
        "Claude-Opus-4.5", "Claude-Sonnet-4.6", "Claude-Sonnet-4.5",
        "Claude-Sonnet-4", "Claude-Haiku-4.5", "Claude-Haiku-4",
        "Claude-3.5-Sonnet"
    ],
    "Google": [
        "Gemini-3.1-Pro", "Gemini-3-Pro", "Gemini-3.5-Flash", "Gemini-2.5-Pro",
        "Gemini-2.5-Flash", "Gemini-2.0-Pro", "Gemini-2.0-Flash", "Gemma-4", "Gemma-3"
    ],
    "Meta": [
        "Llama-4-Maverick", "Llama-4-Scout", "Llama-3.3-70B",
        "Llama-3.1-405B", "Llama-3.1-70B", "Llama-3.2-90B-Vision",
        "Llama-3.2-11B-Vision", "Llama-3.2-3B", "Llama-3.2-1B"
    ],
    "DeepSeek": [
        "DeepSeek-V4-Pro", "DeepSeek-R1", "DeepSeek-R1-Distill",
        "DeepSeek-V3.2", "DeepSeek-V3", "DeepSeek-V2.5"
    ],
    "Alibaba (Qwen)": [
        "Qwen-3.7-Plus", "Qwen-3.5", "Qwen-3", "Qwen-3-Coder",
        "Qwen-2.5-Coder", "Qwen-2.5-72B", "Qwen-2.5-32B", "Qwen-2.5-7B"
    ],
    "Mistral AI": [
        "Mistral-Large-3", "Mistral-Large-2", "Mistral-Medium-3.5",
        "Mistral-Small", "Mistral-7B", "Mixtral-8x22B", "Mixtral-8x7B", "Codestral"
    ],
    "xAI": [
        "Grok-4.3", "Grok-4", "Grok-3", "Grok-2"
    ],
    "Microsoft": [
        "Phi-4", "Phi-4-Multimodal", "Phi-4-Mini", "Phi-3-Medium",
        "Phi-3-Small", "Phi-3-Mini"
    ],
    "Cohere": [
        "Command-A", "Command-R-Plus", "Command-R"
    ],
    "Moonshot AI": [
        "Kimi-K2.7-Code", "Kimi-K2.5", "Kimi-K2", "Kimi-K1"
    ],
    "Z.ai / GLM": [
        "GLM-5.2", "GLM-5", "GLM-4-Plus", "GLM-4-Air", "GLM-4"
    ],
    "NVIDIA": [
        "Nemotron-Ultra", "Nemotron-70B", "Nemotron-4"
    ],
    "Amazon": [
        "Nova-Premier", "Nova-Pro", "Nova-Lite", "Nova-Micro"
    ],
    "IBM": [
        "Granite-3.5", "Granite-3"
    ],
    "AI21": [
        "Jamba-Large", "Jamba-Mini"
    ],
    "01.AI": [
        "Yi-Large", "Yi-Lightning", "Yi-34B"
    ],
    "MiniMax": [
        "MiniMax-M3", "MiniMax-M2", "MiniMax-M1"
    ],
    "Baidu": [
        "ERNIE-4.5", "ERNIE-X1"
    ],
}
