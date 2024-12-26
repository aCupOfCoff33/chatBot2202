# 🧠 AI Expert Chat: Preventing Bullying in Classrooms

This application is an **AI-powered conversational tool** designed to simulate and manage insightful discussions between two "experts" about strategies to prevent bullying in classrooms. It allows users to either observe a live AI-generated discussion or participate as one of the experts, providing actionable insights on this critical topic.

---

## 🚀 Features

### 1. **AI-Driven Expert Conversations**
   - Two AI experts engage in a research-based dialogue about strategies to eliminate bullying.
   - The discussion is generated dynamically using advanced natural language processing (NLP) models.
   - Responses are carefully filtered to avoid repetition and generic statements.

### 2. **Interactive User Mode**
   - Take the role of **Expert 2** and contribute to the conversation.
   - The AI (Expert 1) responds to your input with relevant and actionable strategies.

### 3. **Seamless Mode Switching**
   - Toggle between:
     - **AI-Only Mode**: The two experts converse without user input.
     - **User Interaction Mode**: Participate as Expert 2.

### 4. **Real-Time Chat Interface**
   - Clean and intuitive chat window with:
     - Distinct avatars and styles for both experts.
     - Automatic scrolling to the latest message.

### 5. **Advanced AI Features**
   - **Repetition Avoidance**: Filters out redundant or generic responses.
   - **Semantic Similarity Checks**: Ensures all replies are unique and contextually relevant.
   - **Dynamic Prompting**: Uses a fine-tuned language model (Flan-T5) for intelligent and coherent responses.

---

## 🖼️ Screenshots

1. **Landing Page (Greeting Modal)**  
   *Include a screenshot of the greeting modal explaining the app’s purpose.*

2. **AI-Only Mode Conversation**  
   *Include a screenshot of a conversation between Expert 1 and Expert 2 in AI-only mode.*

3. **User Interaction Mode**  
   *Include a screenshot of the user acting as Expert 2, contributing to the discussion.*

4. **Mode Toggle Button**  
   *Include a screenshot showing the toggle button to switch between AI-only and User Interaction modes.*

---

## 🛠️ Technology Stack

### **Frontend**
- **React.js**: Builds the chat interface and manages user interactions.
- **CSS**: Creates a modern, responsive design.

### **Backend**
- **FastAPI**: Handles AI model interaction and conversation logic.
- **PyTorch**: Runs the language model for response generation.
- **Sentence-Transformers**: Performs semantic similarity checks.

### **AI Models**
- **Flan-T5 Large**: Generates dynamic, research-based responses.
- **SentenceTransformer (all-MiniLM-L6-v2)**: Ensures relevance and uniqueness of responses.

---

## ⚙️ Installation

### Prerequisites
- **Node.js** (for frontend)
- **Python 3.8+** (for backend)
- **pipenv** (optional for Python environment management)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ai-expert-chat.git
   cd ai-expert-chat
   ```
2. **Install backend dependencies**:
   ```bash
   cd api
   pip install -r requirements.txt
   ```
3. **Start the Backend Server**:
   ```bash
   uvicorn chat:app --reload
   ```
4. **Install frontend dependencies**:
   ```bash
   cd ../frontend
   npm install
   ```
5. **Start the frontend server**:
   ```bash
   npm run dev
   ```
6. Access the app:
Open your browser and navigate to http://localhost:5173/

---
## 🤖 How It Works

1. **AI-Only Mode**  
   The backend uses pre-trained NLP models to generate a live discussion between two AI experts. The frontend fetches and displays these responses in real-time.

2. **User Interaction Mode**  
   Users can join as Expert 2 by typing their responses. The backend processes user input and generates a reply from Expert 1, keeping the conversation natural and engaging.

3. **Repetition Avoidance**  
   Responses are filtered to avoid redundancy using semantic similarity checks and a predefined list of “bad responses.”

---

## 🧪 Future Enhancements

- Support for additional discussion topics.
- More diverse AI expert profiles.
- Option to download conversation history as a transcript.
