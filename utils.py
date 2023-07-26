from transformers import AutoTokenizer, pipeline
import torch

def load_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return pipeline(
        "text-generation", 
        model=model_name, 
        tokenizer=tokenizer, 
        device_map="auto",
        torch_dtype=torch.float16
    )

def start_chat(system_prompt):
    return [{"role": "system", "text": system_prompt}]

def construct_system_prompt(sp):
    return f"""
    <s>[INST] <<SYS>>
    {sp}
    <</SYS>>


    """

def convert_history_to_prompt(chat_history):
    prompt = ""
    for i, chat in enumerate(chat_history):
        if chat["role"] == "system":
            prompt += construct_system_prompt(chat["text"])
        elif chat["role"] == "user":
            prompt += f'{chat["text"]} [/INST] '
        elif chat["role"] == "assistant":
            prompt += f'{chat["text"]} </s><s> [INST] '

    return prompt

def extract_model_response(generation):
    return generation[0]["generated_text"].split("[/INST]")[-1].strip()


def continue_chat(user_input, chat_history, model):
    chat_history.append({"role": "user", "text": user_input})
    prompt = convert_history_to_prompt(chat_history)
    print(prompt)
    response = model(prompt, max_length=len(prompt) + 100, num_return_sequences=1)
    model_response = extract_model_response(response)
    chat_history.append({"role": "assistant", "text": model_response})
    return chat_history