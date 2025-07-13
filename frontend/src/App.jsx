import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { ClipLoader } from "react-spinners";
import "./App.css"; // Assuming you have a CSS file for styles
function App() {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState(null);
  const [audio, setAudio] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [pastChats, setPastChats] = useState([]);


  useEffect(() => {
    const handleBeforeUnload = () => {
      if (messages.length > 0) {
        const allSessions = JSON.parse(localStorage.getItem("allChatSessions") || "[]");
        allSessions.push(messages);
        localStorage.setItem("allChatSessions", JSON.stringify(allSessions));
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [messages]);

  const messagesEndRef = useRef(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("chatMessages", JSON.stringify(messages));
    }
  }, [messages]);


  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && !file.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setError("");
    setImage(file);
  };

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file && !file.type.startsWith("audio/")) {
      setError("Please upload a valid audio file.");
      return;
    }
    setError("");
    setAudio(file);
};


  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData();
    formData.append("prompt", prompt);
    if (image) formData.append("image", image);
    if (audio) formData.append("audio", audio);

    try {
      setError("");
      const res = await axios.post(
          "http://127.0.0.1:8000/multi-modal-chat/",
          formData,
         { headers: { "Content-Type": "multipart/form-data" } }
    );
      setMessages((msgs) => [
        ...msgs,
        { role: "user", prompt, image, audio },
        { role: "bot", response: res.data.response },
      ]);
    } catch (err) {
        setError(
          err.response?.data?.error ||
          err.message ||
          "An unexpected error occurred."
      );
}
    setPrompt("");
    setImage(null);
    setAudio(null);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h2>Multi-Modal Chatbot</h2>
      {error && (
        <div style={{ color: "red", marginBottom: 12 }}>
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={prompt}
          onChange={e => {
            setPrompt(e.target.value);
            setError("");
        }}
          placeholder="Type your question..."
          required
          style={{ width: "100%", marginBottom: 8 }}
        />
        <div style={{ marginBottom: 8 }}>
          <label htmlFor="image-input" style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>
            Image:
          </label>
          <input
            id="image-input"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />
        </div>
        {image && (
          <div style={{ marginTop: 8 }}>
            <img
              src={URL.createObjectURL(image)}
              alt="Preview"
              style={{ maxWidth: 200, maxHeight: 200, borderRadius: 8 }}
            />
          </div>
        )}
        <div style={{ marginBottom: 8 }}>
          <label htmlFor="audio-input" style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>
            Audio:
          </label>
          <input
            id="audio-input"
            type="file"
            accept="audio/*"
            onChange={handleAudioChange}
          />
        </div>
        {audio && (
          <div style={{ marginTop: 8 }}>
            <audio controls src={URL.createObjectURL(audio)} />
          </div>
        )}
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Send"}
        </button>
      </form>
      {loading && (
        <div style={{ display: "flex", justifyContent: "center", margin: "16px 0" }}>
         <ClipLoader color="#36d7b7" size={40} />
        </div>
      )}
      
      <button
      onClick={() => {
        const saved = localStorage.getItem("allChatSessions");
        if (saved) setPastChats(JSON.parse(saved));
        setShowHistory(true);
      }}
      style={{ marginBottom: 12 }}
      >
        Show Past Chats
      </button>
      {/* Add the past chats menu here */}
      {showHistory && (
      <div
        style={{
          background: "#f5f5f5",
          border: "1px solid #ccc",
          borderRadius: 8,
          padding: 16,
          marginBottom: 16,
          maxHeight: 300,
          overflowY: "auto",
        }}
      >
        <div style={{ marginBottom: 8, fontWeight: "bold" }}>
          Past Chats
          <button
            onClick={() => setShowHistory(false)}
            style={{ float: "right", fontSize: 14 }}
          >
            Close
          </button>
        </div>
        {pastChats.length === 0 ? (
          <div>No past chats found.</div>
        ) : (
          pastChats.map((session, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <b>Session {i + 1}</b>
              {session.map((msg, idx) =>
                msg.role === "user" ? (
                  <div key={idx} style={{ color: "#00796b" }}>
                    <b>You:</b> {msg.prompt}
                    {msg.image && <div style={{ fontSize: 12 }}>[Image uploaded]</div>}
                    {msg.audio && <div style={{ fontSize: 12 }}>[Audio uploaded]</div>}
                  </div>
                ) : (
                  <div key={idx} style={{ color: "#333" }}>
                    <b>Bot:</b> {msg.response}
                  </div>
                )
              )}
            </div>
          ))
        )}
      </div>
    )}




      <div style={{ marginTop: 24, maxHeight: 400, overflowY: "auto" }}>
        {messages.map((msg, idx) =>
          msg.role === "user" ? (
            <div
              key={idx}
              style={{
              marginBottom: 16,
              background: "#e0f7fa",
              padding: 10,
              borderRadius: 8,
              textAlign: "right",
            }}
          >
          <b>You:</b>
          <div>{msg.prompt}</div>
        {/* Media previews (see below) */}
          {msg.image && (
            <img
              src={URL.createObjectURL(msg.image)}
              alt="User upload"
              style={{ maxWidth: 80, maxHeight: 80, marginTop: 4, borderRadius: 4 }}
            />
          )}
          {msg.audio && (
            <audio controls src={URL.createObjectURL(msg.audio)} style={{ marginTop: 4 }} />
          )}
        </div>
      ) : (
        <div
          key={idx}
          style={{
            marginBottom: 16,
            background: "#f1f8e9",
            padding: 10,
            borderRadius: 8,
            textAlign: "left",
          }}
        >
          <b>Bot:</b>
          <div>{msg.response}</div>
        </div>
      )
    )}
    <div ref={messagesEndRef} />
</div>

    </div>
  );
}

export default App;
