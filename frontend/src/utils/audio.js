// Audio utility functions
// Note: FFmpeg conversion is no longer needed as the backend 
// now handles all audio format conversions automatically

export function createAudioFile(blob, filename = 'audio.webm') {
  return new File([blob], filename, {
    type: blob.type || 'audio/webm',
    lastModified: Date.now()
  });
}

export function getAudioInfo(file) {
  return {
    name: file.name,
    size: file.size,
    type: file.type,
    lastModified: file.lastModified
  };
}