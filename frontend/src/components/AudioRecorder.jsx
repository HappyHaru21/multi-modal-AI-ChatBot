import React, { useState } from "react";
import { ReactMediaRecorder } from "react-media-recorder";
import { safeConvertToMp3 } from "../utils/mp3Converter"; // Adjust the path if needed

const AudioRecorder = ({ onAudioReady }) => {
  const [mediaBlobUrl, setMediaBlobUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");

    const handleStop = async (blobUrl, blob) => {
    setMediaBlobUrl(blobUrl);
    setError("");
    setIsProcessing(true);
    try {
        const mp3File = await safeConvertToMp3(blob);
        if (mp3File.type !== "audio/mp3" && mp3File.type !== "audio/mpeg") {
        setError("Audio conversion failed. Please try a different browser (Chrome recommended) or upload an MP3/WAV file.");
        onAudioReady(null);
        return;
        }
        onAudioReady(mp3File);
    } catch (error) {
        setError("Audio conversion failed. Please try again or upload a compatible audio file.");
        onAudioReady(null);
    } finally {
        setIsProcessing(false);
    }
    };

  const clearRecording = () => {
    setMediaBlobUrl(null);
    setError("");
    onAudioReady(null);
  };

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ marginBottom: 8, fontWeight: "bold" }}>
        Record Audio:
      </div>
      {error && <div style={{ color: 'red', marginBottom: 8 }}>{error}</div>}
      <ReactMediaRecorder
        audio={{
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }}
        onStop={handleStop}
        render={({ status, startRecording, stopRecording }) => (
          <div>
            <div style={{ marginBottom: 8, fontSize: 14, color: '#666' }}>
              Status: {status} {isProcessing && "(Converting to MP3...)"}
            </div>
            <div style={{ marginBottom: 8 }}>
              <button
                type="button"
                onClick={startRecording}
                disabled={status === "recording"}
                style={{ 
                  marginRight: 8,
                  backgroundColor: status === "recording" ? '#ccc' : '#4CAF50',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: 4,
                  cursor: status === "recording" ? 'not-allowed' : 'pointer'
                }}
              >
                🎤 Start Recording
              </button>
              <button
                type="button"
                onClick={stopRecording}
                disabled={status !== "recording"}
                style={{
                  marginRight: 8,
                  backgroundColor: status !== "recording" ? '#ccc' : '#f44336',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: 4,
                  cursor: status !== "recording" ? 'not-allowed' : 'pointer'
                }}
              >
                ⏹️ Stop Recording
              </button>
              {mediaBlobUrl && (
                <button
                  type="button"
                  onClick={clearRecording}
                  style={{
                    backgroundColor: '#ff9800',
                    color: 'white',
                    border: 'none',
                    padding: '8px 16px',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  🗑️ Clear
                </button>
              )}
            </div>
            {mediaBlobUrl && (
              <div style={{ marginTop: 8 }}>
                <audio src={mediaBlobUrl} controls style={{ width: '100%' }} />
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                  Audio ready to send as MP3! Click "Send" to process.
                </div>
              </div>
            )}
          </div>
        )}
      />
    </div>
  );
};

export default AudioRecorder;
