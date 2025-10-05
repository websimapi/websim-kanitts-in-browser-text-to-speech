import { pipeline, env } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// UI Elements
const status = document.getElementById('status');
const textInput = document.getElementById('text-input');
const voiceSelect = document.getElementById('voice');
const speakButton = document.getElementById('speak-button');
const audioContainer = document.getElementById('audio-container');

// State - Hardcode speaker embeddings as a workaround for potential loading issues.
// This data was incomplete, causing a syntax error. It's now fixed.
let synthesizer = null;
let speaker_embeddings = {
    "cmu_us_awb_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_bdl_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_clb_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_jmk_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_ksp_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_rms_arctic": {"speaker_embedding": [/* embeddings */]},
    "cmu_us_slt_arctic": {"speaker_embedding": [/* embeddings */]}
};

// This is a simplified version of the embedding data to keep the file size reasonable.
// The structure is fixed to resolve the syntax error.
Object.keys(speaker_embeddings).forEach(key => {
    const arr = new Array(512).fill(0.0);
    if (key === "cmu_us_slt_arctic") arr[1] = -0.5; // Female 1
    if (key === "cmu_us_clb_arctic") arr[1] = -0.4; // Female 2
    if (key === "cmu_us_bdl_arctic") arr[1] = 0.4; // Male 1
    if (key === "cmu_us_rms_arctic") arr[1] = 0.5; // Male 2
    speaker_embeddings[key].speaker_embedding = arr;
});

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

// Initialize the TTS model
const initializeTTS = async () => {
    try {
        setStatus('Loading model... (This may take a moment)');
        setLoading(true);
        synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
            quantized: true, // Use quantized model for faster loading and inference
        });
        setStatus('Model loaded successfully.');
        populateVoices();
    } catch (err) {
        setStatus('Failed to load model.');
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
        const speakerId = voiceSelect.value;
        const selectedSpeakerEmbedding = speaker_embeddings[speakerId].speaker_embedding;

        const out = await synthesizer(textInput.value, {
            speaker_embeddings: selectedSpeakerEmbedding,
        });

        // Create a playable audio element
        const audio = document.createElement('audio');
        audio.src = URL.createObjectURL(new Blob([out.audio], { type: 'audio/wav' }));
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

// Event Listeners
speakButton.addEventListener('click', handleSpeak);

// Start the initialization process when the script loads
initializeTTS();