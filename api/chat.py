# api/chat.py
import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from mangum import Mangum  # Import Mangum

# Initialize the FastAPI application
app = FastAPI()

# Configure CORS to allow only your frontend's Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.vercel.app"],  # Replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (e.g., GET, POST)
    allow_headers=["*"],  # Allow all headers
)

# Define phrases that should be filtered out as bad responses
BAD_RESPONSES = [
    "",  # Empty string
    " ",  # Blank response
    "Expert 1: ",  # Placeholder with no content
    "Expert 2: ",  # Placeholder with no content
    "This is a great idea.",
    "That's an excellent point.",
    "That's great!",
    "Good idea.",
    "Good point.",
    "I agree.",
    "Thank you.",
    "Certainly.",
    "Absolutely.",
    "Indeed.",
    "Sure.",
    "Of course.",
    "That's true.",
    "You're right.",
    "I think so too.",
    "Let's continue.",
    "That's helpful.",
    "I would agree.",
    "I would agree with you.",
    "great idea",
    "fantastic idea",
    "good idea",
    "excellent idea",
    "wonderful idea",
    "amazing idea",
    "superb idea",
    "thanks for your time",
    "thank you for your time",
    "I would agree",
    "Thanks for the information.",
]

# Determine the device for running the model
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the pre-trained language model (flan-t5-large)
MODEL_NAME = "google/flan-t5-large"

# Initialize the tokenizer for encoding/decoding text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the sentence-transformers model for semantic similarity checks
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the main transformer model with appropriate device settings
if device.type in ["mps", "cuda"]:  # Use GPU if available
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,  # Use half-precision for efficiency
        low_cpu_mem_usage=True
    )
else:  # Use CPU otherwise
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use full precision on CPU
        low_cpu_mem_usage=True
    ).to(device)
print("Model loaded successfully.")

def is_repeated_idea(response, history):
    """
    Check if the newly generated response is semantically similar to any prior responses in the chat history.
    """
    # Extract the text content from the history
    history_lines = [line.split(": ", 1)[1] for line in history.split("\n") if ": " in line]
    if not history_lines:  # If no history exists, no repetition
        return False

    # Encode history and response into embeddings
    history_embeddings = similarity_model.encode(history_lines, convert_to_tensor=True)
    response_embedding = similarity_model.encode(response, convert_to_tensor=True)

    # Compute similarity scores between the response and all historical responses
    similarities = util.pytorch_cos_sim(response_embedding, history_embeddings)
    max_similarity = similarities.max().item()

    # Consider the response repeated if the maximum similarity exceeds 0.8
    if max_similarity > 0.8:
        print(f"Rejected response as repeated idea (similarity={max_similarity:.2f}): {response}")
        return True
    return False

def generate_response(prompt, max_length=150, temperature=0.8, top_p=0.9, no_repeat_ngram_size=3):
    """
    Generate a response from the model given a prompt, using specific parameters for diversity.
    """
    # Tokenize the prompt and prepare inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(model.device)

    # Generate the response using model parameters
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,  # Increase randomness for creativity
        do_sample=True,
        top_p=top_p,  # Focus on the top tokens for sampling
        no_repeat_ngram_size=no_repeat_ngram_size,  # Avoid repeated phrases
        repetition_penalty=1.2,  # Penalize repetitive text
        eos_token_id=tokenizer.eos_token_id,  # Ensure response ends at the end-of-sequence token
    )
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

class ChatHistory(BaseModel):
    """
    Pydantic model for validating the chat history received in POST requests.
    """
    history: str  # The conversation history
    mode: Optional[str] = "experts"  # The current mode (default: "experts")

def determine_current_speaker(history: str, mode: str) -> Optional[str]:
    """
    Determine the next speaker (Expert 1 or Expert 2) based on the conversation history.
    """
    last_line = history.strip().split("\n")[-1]  # Get the last line of the history
    if mode == "experts":
        if last_line.startswith("Expert 1:"):
            return "Expert 2"  # Switch to Expert 2
        elif last_line.startswith("Expert 2:"):
            return "Expert 1"  # Switch to Expert 1
        else:
            return "Expert 1"  # Default to Expert 1
    elif mode == "expert2":
        if last_line.startswith("Expert 2:"):
            return "Expert 1"
        elif last_line.startswith("Expert 1:"):
            return None  # Wait for Expert 2 (user) to send a message
        else:
            return "Expert 1"  # Default to Expert 1
    else:
        return None  # Invalid mode

@app.post("/chat")
def chat(data: ChatHistory):
    """
    Handle chat requests. Generate responses from the appropriate expert.
    """
    print("Request received.")
    history = data.history
    mode = data.mode
    print(f"History:\n{history}")
    print(f"Mode: {mode}")

    # Determine the current speaker
    current_speaker = determine_current_speaker(history, mode)
    print(f"Current speaker: {current_speaker}")

    if current_speaker is None:
        print("No speaker to respond. Returning empty response.")
        return {"response": ""}

    # Set the instruction based on the mode
    if mode == "experts":
        instruction = (
            "The following is a conversation between Expert 1 and Expert 2 about strategies to eliminate bullying in classrooms. "
            "Both experts specialize in educational psychology and student welfare. "
            "They provide actionable, research-based insights, avoiding repetition or generic statements."
        )
    elif mode == "expert2":
        instruction = (
            "The following is a conversation between Expert 1 and Expert 2 about strategies to eliminate bullying in classrooms. "
            "Expert 1 is an AI expert in educational psychology, while Expert 2 is a human with experience in student welfare. "
            "Only Expert 1 generates responses, focusing on clear, actionable strategies."
        )
    else:
        print("Invalid mode provided. Returning empty response.")
        return {"response": ""}

    # Prepare the prompt for the model
    prompt = f"{instruction}\n{history}\n{current_speaker}:"
    print(f"Prompt:\n{prompt}")

    used_responses = set()  # Track responses generated in this session

    # Attempt to generate a valid response
    for attempt in range(5):
        response_text = generate_response(prompt)
        response_clean = response_text.strip()

        # Log the generated response
        print(f"Generated response (attempt {attempt + 1}): {response_clean}")

        # Check for invalid or repeated responses
        if response_clean in BAD_RESPONSES or is_repeated_idea(response_clean, history):
            print(f"Rejected response: {response_clean}")
            continue

        # Avoid duplicate responses in this session
        if response_clean in used_responses:
            print(f"Rejected duplicate response: {response_clean}")
            continue

        # Accept the response if it passes all checks
        used_responses.add(response_clean)
        break
    else:
        # Provide a fallback response if all attempts fail
        response_clean = (
            f"{current_speaker}: Another proven method involves empowering bystanders to report bullying and take an active role in supporting victims."
        )

    print(f"Returning response: {response_clean}")
    return {"response": response_clean}

# Remove signal handling as it's not needed in serverless

# Remove the traditional server run
# if __name__ == "__main__":
#     uvicorn.run("chat:app", host="0.0.0.0", port=8000, reload=False)

# Add Mangum handler for Vercel
handler = Mangum(app)