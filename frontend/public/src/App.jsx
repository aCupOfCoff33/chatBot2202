// frontend/src/App.jsx
import React, { useState, useEffect, useRef } from "react";

const App = () => {
  // State to hold the conversation messages
  const [messages, setMessages] = useState([
    {
      sender: "Expert 1",
      text: "One effective strategy to prevent school bullying is implementing comprehensive anti-bullying policies that are clearly communicated to students, parents, and staff. This ensures everyone understands what constitutes bullying and the consequences of such behavior.",
    },
    {
      sender: "Expert 2",
      text: "I agree. Additionally, schools should provide regular training for teachers to recognize and intervene in bullying situations effectively. Educators equipped with the right tools can make a significant difference.",
    },
  ]);

  // State to track if a fetch operation is ongoing
  const [isFetching, setIsFetching] = useState(false);

  // State to store the user's input message
  const [inputValue, setInputValue] = useState("");

  // State to determine the current mode (AI conversation or user participation as Expert 2)
  const [mode, setMode] = useState("experts"); // "experts" or "expert2"

  // State to manage the visibility of the greeting modal
  const [showModal, setShowModal] = useState(true);

  // Reference to enable automatic scrolling to the latest message
  const messagesEndRef = useRef(null);

  // Function to scroll to the latest message in the chat window
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Automatically scroll to the bottom whenever messages are updated
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to fetch the next AI-generated message from the backend
  const fetchNextMessage = async (currentHistory) => {
    if (isFetching) return; // Prevent overlapping fetches
    setIsFetching(true);

    try {
      console.log("Sending history to backend:", currentHistory);
      console.log("Current mode:", mode);

      // Make a POST request to the backend API
      const response = await fetch(import.meta.env.VITE_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ history: currentHistory, mode }),
      });

      // Handle unsuccessful responses
      if (!response.ok) {
        console.error(`Server error: ${response.status}`);
        throw new Error("Failed to fetch message from the backend.");
      }

      const data = await response.json();
      console.log("Backend response:", data);

      // Handle errors in the backend response
      if (data.error) {
        console.error("Backend error:", data.error);
        setMessages((prev) => [
          ...prev,
          { sender: "Error", text: data.error },
        ]);
        return;
      }

      // Extract and add the new message to the chat
      if (data.response) {
        const [speaker, ...messageParts] = data.response.split(":");
        const message = messageParts.join(":").trim();

        console.log("New message:", message);
        setMessages((prev) => [
          ...prev,
          { sender: speaker.trim(), text: message },
        ]);
      }
    } catch (error) {
      console.error("Fetch error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "Error", text: "Failed to fetch message. Try again later." },
      ]);
    } finally {
      setIsFetching(false); // Reset fetching status
    }
  };

  // Function to handle sending messages from Expert 2
  const handleSendMessage = () => {
    if (inputValue.trim() === "") return; // Do nothing if the input is empty

    const sender = "Expert 2"; // Define the sender as Expert 2 (user)
    const newMessage = { sender, text: inputValue };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);

    // Compute the new conversation history
    const newHistory = updatedMessages
      .map((msg) => `${msg.sender}: ${msg.text}`)
      .join("\n");

    // Fetch the next response from the backend in 'expert2' mode
    if (mode === "expert2") {
      fetchNextMessage(newHistory);
    }

    setInputValue(""); // Clear the input field
  };

  // Handle Expert-to-Expert conversations in 'experts' mode
  useEffect(() => {
    if (!showModal && mode === "experts") {
      const interval = setInterval(() => {
        const currentHistory = messages
          .map((msg) => `${msg.sender}: ${msg.text}`)
          .join("\n");
        fetchNextMessage(currentHistory); // Fetch the next response every 5 seconds
      }, 5000);

      return () => clearInterval(interval); // Cleanup the interval on unmount
    }
  }, [mode, showModal, messages]);

  // Function to toggle between 'experts' and 'expert2' modes
  const toggleMode = () => {
    if (mode === "experts") {
      setMode("expert2"); // Switch to Expert Interaction
    } else if (mode === "expert2") {
      setMode("experts"); // Switch back to AI-only conversation
    }
  };

  // Function to handle pressing the Enter key in the input field
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSendMessage(); // Send the message on Enter
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-900 text-white">
      {/* Greeting Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-11/12 max-w-md relative">
            {/* Close Button */}
            <button
              className="absolute top-2 right-2 text-gray-400 hover:text-gray-200"
              onClick={() => setShowModal(false)} // Close the modal
            >
              &times;
            </button>
            <h2 className="text-2xl font-bold mb-4">Welcome!</h2>
            <p className="mb-6">
              This application features an AI-generated conversation between
              experts discussing strategies to prevent bullying. We've initiated
              the chat with two default messages to set the stage. You have the
              option to let the AI experts continue the discussion or to join in
              yourself. Simply use the toggle to switch modes. If you choose to
              participate, you'll become <strong>Expert 2</strong>!
            </p>
            <button
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 w-full"
              onClick={() => setShowModal(false)} // Close the modal
            >
              OK
            </button>
          </div>
        </div>
      )}

      <div className="flex flex-col w-[90%] max-w-4xl h-[80%] max-h-screen bg-gray-800 rounded-lg shadow-lg p-6">
        {/* Header with Title and Toggle Button */}
        <div className="flex justify-between items-center mb-4">
          {/* Header Title */}
          <h1 className="text-2xl font-bold">Dr. Yimin Yang GPT</h1>

          {/* Toggle Mode Button */}
          <button
            onClick={toggleMode}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            {mode === "experts"
              ? "Switch to Expert Interaction"
              : "Switch to Expert Mode"}
          </button>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 border border-gray-700 rounded-lg">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.sender === "Expert 1"
                  ? "justify-start items-start"
                  : "justify-end items-end"
              } my-3`}
            >
              {/* Expert 1 Avatar */}
              {msg.sender === "Expert 1" && (
                <img
                  src="/src/images/prof.JPG"
                  alt="Expert 1"
                  className="w-10 h-10 rounded-full mr-2"
                />
              )}

              {/* Message Bubble */}
              <div
                className={`px-4 py-2 rounded-lg text-sm ${
                  msg.sender === "Expert 1"
                    ? "bg-blue-600 text-white"
                    : "bg-green-600 text-white"
                }`}
              >
                <strong>{msg.sender}: </strong>
                {msg.text}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} /> {/* Reference for auto-scroll */}
        </div>

        {/* Input Field for Expert 2 */}
        {mode === "expert2" && (
          <div className="mt-4 flex items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)} // Update input state
              placeholder="Type your message here..."
              className="flex-1 px-4 py-2 rounded-l-lg bg-gray-700 text-white border border-gray-600 focus:outline-none"
              onKeyDown={handleKeyDown} // Handle Enter key
            />
            <button
              onClick={handleSendMessage}
              disabled={isFetching} // Disable button while fetching
              className="px-4 py-2 rounded-r-lg bg-blue-600 text-white border border-blue-500 hover:bg-blue-700 focus:outline-none disabled:opacity-50"
            >
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;