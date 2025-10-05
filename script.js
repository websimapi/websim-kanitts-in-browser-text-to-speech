import { pipeline, env } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;
// Use the wasm backend for performance
env.backends.onnx.wasm.numThreads = 1;

// UI Elements
const status = document.getElementById('status');
const textInput = document.getElementById('text-input');
const voiceSelect = document.getElementById('voice');
const speakButton = document.getElementById('speak-button');
const audioContainer = document.getElementById('audio-container');
const voiceUpload = document.getElementById('voice-upload');

// State
let synthesizer = null;
let speaker_embeddings = null;
let embeddingExtractor = null;
let customSpeakerEmbedding = null;

// Helper function to decode and resample audio using Web Audio API
async function decodeAndResampleAudio(file, targetSr) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();

    // Use a promise to handle the callback-based decodeAudioData
    const decodedAudio = await new Promise((resolve, reject) => {
        audioContext.decodeAudioData(arrayBuffer, resolve, reject);
    });

    const originalSr = decodedAudio.sampleRate;

    if (originalSr === targetSr) {
        // No resampling needed, return the first channel data
        return { data: decodedAudio.getChannelData(0), sampling_rate: originalSr };
    }

    // Resample using OfflineAudioContext
    const duration = decodedAudio.duration;
    // The speaker encoder expects single-channel audio
    const offlineContext = new OfflineAudioContext(1, duration * targetSr, targetSr);
    const bufferSource = offlineContext.createBufferSource();
    bufferSource.buffer = decodedAudio;
    bufferSource.connect(offlineContext.destination);
    bufferSource.start();

    const resampledAudio = await offlineContext.startRendering();
    // Return the first channel data
    return { data: resampledAudio.getChannelData(0), sampling_rate: targetSr };
}

// Helper function to convert raw PCM audio data to a WAV file format
function pcmToWav(pcm, sampleRate) {
    const numSamples = pcm.length;
    const numChannels = 1;
    const bytesPerSample = 2; // 16-bit audio
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * bytesPerSample;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true); // file size - 8
    writeString(view, 8, 'WAVE');

    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // sub-chunk size
    view.setUint16(20, 1, true);  // audio format (1 for PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample

    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write PCM data
    let offset = 44;
    for (let i = 0; i < pcm.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, pcm[i]));
        // Convert to 16-bit signed integer
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
}

// Function to update the UI status
const setStatus = (text) => {
    status.textContent = text;
};

// Function to toggle loading state on the button
const setLoading = (isLoading) => {
    if (isLoading) {
        speakButton.classList.add('loading');
        speakButton.disabled = true;
    } else {
        speakButton.classList.remove('loading');
        speakButton.disabled = false;
    }
};

// Initialize the TTS models
const initializeTTS = async () => {
    try {
        setStatus('Loading models... (This may take a moment)');
        setLoading(true);

        // Load the TTS synthesizer
        synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
            quantized: true,
        });

        // Load the speaker embedding extractor
        embeddingExtractor = await pipeline('feature-extraction', 'Xenova/speecht5_speaker-encoder', {
            quantized: true,
        });

        // Load pre-defined speaker embeddings
        const speaker_embeddings_url = 'https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/speaker_embeddings.json';
        const response = await fetch(speaker_embeddings_url);
        speaker_embeddings = await response.json();

        setStatus('Models loaded successfully.');
        populateVoices();
    } catch (err) {
        setStatus('Failed to load models.');
        console.error('Initialization error:', err);
    } finally {
        setLoading(false);
    }
};

// Populate the voice selector dropdown
const populateVoices = () => {
    const voiceOptions = Object.keys(speaker_embeddings).map(speakerId => {
        // Create more user-friendly names
        const name = speakerId.replace('cmu_us_', '').replace('_arctic', '').toUpperCase();
        let gender = 'Male';
        if (speakerId.includes('clb') || speakerId.includes('slt')) {
            gender = 'Female';
        }
        return { id: speakerId, name: `${gender} ${name}` };
    });

    voiceSelect.innerHTML = '';
    voiceOptions.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.id;
        option.textContent = voice.name;
        voiceSelect.appendChild(option);
    });
};

// Function to handle the speech synthesis
const handleSpeak = async () => {
    if (!synthesizer || !textInput.value) {
        return;
    }

    setLoading(true);
    setStatus('Synthesizing audio...');
    audioContainer.innerHTML = '';

    try {
        let speakerEmbedding;

        if (customSpeakerEmbedding) {
             // Use the custom uploaded voice
            speakerEmbedding = customSpeakerEmbedding;
            setStatus('Synthesizing with custom voice...');
        } else {
            // Use the selected voice from the dropdown
            const speakerId = voiceSelect.value;
            speakerEmbedding = speaker_embeddings[speakerId];
        }

        const out = await synthesizer(textInput.value, {
            speaker_embeddings: speakerEmbedding,
        });

        // Convert the raw PCM audio to a WAV file
        const wavBuffer = pcmToWav(out.audio, out.sampling_rate);
        
        // Create a playable audio element
        const audio = document.createElement('audio');
        audio.src = URL.createObjectURL(new Blob([wavBuffer], { type: 'audio/wav' }));
        audio.controls = true;
        audio.autoplay = true;
        audioContainer.appendChild(audio);

        setStatus('Done.');
    } catch (err) {
        setStatus('An error occurred during synthesis.');
        console.error('Synthesis error:', err);
    } finally {
        setLoading(false);
    }
};

// Function to handle uploaded voice file
const handleVoiceUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
        customSpeakerEmbedding = null;
        setStatus('Custom voice cleared.');
        return;
    }

    try {
        setLoading(true);
        setStatus('Processing custom voice...');

        // The model expects audio with a sampling rate of 16000
        const audio = await decodeAndResampleAudio(file, 16000);

        // Extract speaker embeddings
        const embeddings = await embeddingExtractor(audio.data, {
            sampling_rate: audio.sampling_rate,
        });

        // The feature-extraction pipeline returns the embedding tensor directly.
        customSpeakerEmbedding = embeddings.data;
        
        setStatus('Custom voice loaded successfully. Press "Speak" to use it.');
        // Deselect any option in the dropdown to indicate custom voice is active
        voiceSelect.value = ''; 
    } catch (err) {
        setStatus('Failed to process custom voice.');
        console.error('Voice processing error:', err);
        customSpeakerEmbedding = null;
    } finally {
        setLoading(false);
    }
};

// Event Listeners
speakButton.addEventListener('click', handleSpeak);
voiceUpload.addEventListener('change', handleVoiceUpload);
voiceSelect.addEventListener('change', () => {
    // When a preset voice is selected, clear the custom one
    if (voiceUpload.value) {
        voiceUpload.value = ''; // Reset file input
    }
    customSpeakerEmbedding = null;
    setStatus('Switched to preset voice.');
});

// Start the initialization process when the script loads
initializeTTS();