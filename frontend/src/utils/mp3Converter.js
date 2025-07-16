import lamejs from 'lamejs';

/**
 * Converts audio blob to MP3 format
 * @param {Blob} audioBlob - The audio blob to convert
 * @returns {Promise<File>} - Promise that resolves to MP3 file
 */
export const convertToMp3 = async (audioBlob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = async (event) => {
      try {
        const arrayBuffer = event.target.result;
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Decode the audio data
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Get the audio data as Float32Array
        const leftChannel = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        
        // Convert Float32Array to Int16Array (required by lamejs)
        const samples = new Int16Array(leftChannel.length);
        for (let i = 0; i < leftChannel.length; i++) {
          // Convert from -1.0 to 1.0 range to -32768 to 32767 range
          samples[i] = Math.max(-32768, Math.min(32767, leftChannel[i] * 32768));
        }
        
        // Initialize MP3 encoder
        const mp3encoder = new lamejs.Mp3Encoder(1, sampleRate, 128); // mono, sampleRate, 128kbps
        
        // Encode to MP3
        const mp3Data = [];
        const sampleBlockSize = 1152; // MP3 frame size
        
        for (let i = 0; i < samples.length; i += sampleBlockSize) {
          const sampleChunk = samples.subarray(i, i + sampleBlockSize);
          const mp3buf = mp3encoder.encodeBuffer(sampleChunk);
          if (mp3buf.length > 0) {
            mp3Data.push(mp3buf);
          }
        }
        
        // Flush remaining data
        const finalBuffer = mp3encoder.flush();
        if (finalBuffer.length > 0) {
          mp3Data.push(finalBuffer);
        }
        
        // Create blob from MP3 data
        const mp3Blob = new Blob(mp3Data, { type: 'audio/mp3' });
        
        // Create file with .mp3 extension
        const mp3File = new File([mp3Blob], 'recording.mp3', {
          type: 'audio/mp3',
          lastModified: Date.now()
        });
        
        resolve(mp3File);
        
      } catch (error) {
        console.error('MP3 conversion error:', error);
        reject(error);
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read audio blob'));
    };
    
    reader.readAsArrayBuffer(audioBlob);
  });
};

/**
 * Converts audio blob to MP3 with error handling and fallback
 * @param {Blob} audioBlob - The audio blob to convert
 * @returns {Promise<File>} - Promise that resolves to MP3 file or original file if conversion fails
 */
export const safeConvertToMp3 = async (audioBlob) => {
  try {
    console.log('Starting MP3 conversion...');
    const mp3File = await convertToMp3(audioBlob);
    console.log('MP3 conversion successful:', mp3File.name, mp3File.size, 'bytes');
    return mp3File;
  } catch (error) {
    console.warn('MP3 conversion failed, using original format:', error);
    // Fallback: return original blob as file
    return new File([audioBlob], 'recording.webm', {
      type: audioBlob.type || 'audio/webm',
      lastModified: Date.now()
    });
  }
};
