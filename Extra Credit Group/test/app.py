import gradio as gr
# Removed Unsloth import
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel # Added PEFT import
import torch
import socket
import os
import threading
from queue import Queue

# --- Configuration ---
# Point to the repo with LoRA adapters, NOT the GGUF repo
ADAPTER_MODEL_NAME = "jagruthh/home_remedies_model"
# Specify the original base model the adapters were trained on
# (Found in the adapter_config.json of the ADAPTER_MODEL_NAME repo)
BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

MAX_SEQ_LENGTH = 2048
# LOAD_IN_4BIT = True # Removed - Not typically used for CPU / standard HF

# Global variables
model = None
tokenizer = None

def find_available_port(start_port=7860, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError("No available ports found.")

def load_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"Loading base model: {BASE_MODEL_NAME}...")
        # Load base model using standard Hugging Face transformers
        # Use float32 for CPU execution
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32, # Use float32 for CPU
            # device_map="auto" # Usually fine for CPU, loads to RAM
        )
        print(f"Loading LoRA adapters: {ADAPTER_MODEL_NAME}...")
        # Apply the LoRA adapters using PEFT
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_NAME)
        model.eval() # Set model to evaluation mode

        print("Loading tokenizer...")
        # Load tokenizer associated with the adapters/fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_NAME)

        # FastLanguageModel.for_inference(model) # Removed - Unsloth specific

        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        print("Model and tokenizer loaded for CPU execution.")
    return model, tokenizer

# GradioTextStreamer remains the same as it's independent of Unsloth/GPU
class GradioTextStreamer(TextStreamer):
    def __init__(self, tokenizer, queue: Queue, skip_prompt: bool = True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.queue = queue
        self.stop_signal = None # Using None is simpler for Queue sentinel
        self.timeout = 120

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Make sure not to put empty strings from initial prompt processing
        if text:
             self.queue.put(text, timeout=self.timeout)
        if stream_end:
            self.queue.put(self.stop_signal, timeout=self.timeout) # Signal end

# --- Prompt ---
REMEDY_INSTRUCTION = (
    "You are a home remedy expert. Suggest a safe, natural remedy using common household ingredients "
    "like honey, ginger, turmeric, etc. Do NOT suggest prescription medication. Keep your response under 80 words."
)

@torch.inference_mode()
def generate_remedy(illness):
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            load_model_and_tokenizer()
        except Exception as e:
             print(f"Error loading model during generation: {e}")
             yield f"Error: Failed to load model - {e}"
             return
        if model is None or tokenizer is None:
            yield "Error: Model not loaded after attempt."
            return

    input_text = f"Illness: {illness}"
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{REMEDY_INSTRUCTION}

### Input:
{input_text}

### Response:
"""
    # Prepare inputs - DO NOT move to CUDA
    inputs = tokenizer([prompt], return_tensors="pt") # Keep on CPU
    q = Queue()
    streamer = GradioTextStreamer(tokenizer, q, skip_prompt=True)

    def generation_thread_func():
        try:
            # Generate on CPU
            model.generate(
                **inputs, # Pass inputs dictionary directly
                streamer=streamer,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error during generation: {e}") # Log error
            # Ensure queue gets the error and stop signal
            try:
                q.put(f"\n\nGeneration Error: {e}", timeout=5)
            except Queue.Full:
                pass
            try:
                q.put(streamer.stop_signal, timeout=5) # Use stop_signal correctly
            except Queue.Full:
                 pass # Avoid blocking if queue is full

    thread = threading.Thread(target=generation_thread_func)
    thread.start()

    generated_text = ""
    while True:
        try:
            next_token = q.get(timeout=120) # Wait up to 120 seconds
            if next_token is streamer.stop_signal: # Check for sentinel
                break
            generated_text += next_token
            # Strip known EOS tokens if they appear mid-stream
            # Also check for potential duplicated text which can happen sometimes
            clean_text = generated_text.strip().replace("<|eot_id|>", "").replace("</s>", "").strip()
            yield clean_text # Yield the cleaned text to Gradio

        except Queue.Empty:
            # Check if thread died unexpectedly
            if not thread.is_alive():
                print("Generation thread finished unexpectedly.")
                break
            # Otherwise, just continue waiting for the queue
            continue
        except Exception as e:
            print(f"Error reading from queue: {e}")
            break # Exit loop on other queue errors

    # Wait for the thread to finish, with a timeout
    thread.join(timeout=10)
    if thread.is_alive():
        print("Warning: Generation thread did not finish cleanly after timeout.")


# --- Load model eagerly before starting UI ---
print("Attempting to load model before starting Gradio...")
try:
    load_model_and_tokenizer()
    print("Pre-loading successful.")
except Exception as e:
    print(f"Error during pre-loading: {e}")
    # Decide if you want to proceed without a loaded model or exit
    # For now, we'll let Gradio handle the loading attempt later if needed

# Illness examples (same as before)
examples = [
    ["headache"], ["cough"], ["indigestion"], ["sore throat"],
    ["constipation"], ["nausea"], ["sunburn"], ["muscle cramps"],
]

# CSS for UI (same as before)
css = """
    .gradio-container { font-family: 'Segoe UI', sans-serif; }
    .feedback-box textarea {
        background-color: #fefcf9; border: 1px solid #ddd; padding: 12px;
        border-radius: 10px; color: #2e2e2e; font-size: 16px;
    }
    .title { font-size: 32px; color: #2f855a; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .section-title { font-size: 20px; margin-bottom: 10px; color: #4a5568; font-weight: 600; }
    .gr-button-primary { background-color: #38a169 !important; color: white !important; }
"""

# --- Gradio UI (mostly same as before) ---
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Image("SLU.png", elem_id="uni-logo", width="100%")
            
    gr.Markdown("<div class='title'>🌿 Natural Home Remedies Generator (CPU)</div>") # Updated title
    gr.Markdown(
        "<hr><p style='text-align: center; font-size: 16px; color: #555;'>"
        "⚠️ Please note: This app may take 2–5 minutes to respond, as the language model is running on a CPU instead of a GPU."
        "</p>"
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>🩺 What’s the illness?</div>")
            illness_input = gr.Textbox(
                placeholder="e.g., headache, sore throat, indigestion...",
                lines=2,
                show_label=False
            )
            submit_btn = gr.Button("🌼 Suggest Remedy", variant="primary")
            gr.Markdown("<hr><div class='section-title'>🧪 Try these:</div>")
            gr.Examples(
                examples=examples,
                inputs=[illness_input],
                label="Sample Illnesses",
            )

        with gr.Column(scale=2):
            gr.Markdown("<div class='section-title'>💡 Home Remedy</div>")
            remedy_output = gr.Textbox(
                label=None,
                interactive=False,
                lines=8,
                elem_classes="feedback-box"
            )

    submit_btn.click(
        fn=generate_remedy,
        inputs=[illness_input],
        outputs=remedy_output,
        api_name="remedy"
    )

# --- Launch ---
if __name__ == "__main__":
    try:
        port = find_available_port()
        print(f"✅ Launching on port: {port}")
        print("🐌 NOTE: CPU inference will be significantly slower than GPU.")
        print("💾 NOTE: Loading the full model requires substantial RAM.")
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=True # Set to False if you don't need public sharing
        )
    except Exception as e:
        print(f"❌ Error launching app: {e}")