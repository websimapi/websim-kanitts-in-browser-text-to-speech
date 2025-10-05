import { pipeline, env } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// UI Elements
const status = document.getElementById('status');
const textInput = document.getElementById('text-input');
const voiceSelect = document.getElementById('voice');
const speakButton = document.getElementById('speak-button');
const audioContainer = document.getElementById('audio-container');

// State
let synthesizer = null;
let speaker_embeddings = null;
const speaker_embeddings_url = 'https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors/resolve/main/speaker_embeddings.json';

// Function to update status messages
const setStatus = (message) => {
    status.textContent = message;
};

// Function to initialize the synthesizer
const initializeSynthesizer = async () => {
    setStatus('Loading model and voices... this may take a moment.');
    speakButton.disabled = true;
    speakButton.classList.add('loading');

    try {
        synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
            quantized: false,
        });

        const speakerEmbeddingsResponse = await fetch(speaker_embeddings_url);
        speaker_embeddings = await speakerEmbeddingsResponse.json();

        // Populate voice selector
        Object.keys(speaker_embeddings).forEach(speakerId => {
            const option = document.createElement('option');
            option.value = speakerId;
            option.textContent = speakerId.replace('cmu_us_', '').replace('_arctic', '').toUpperCase();
            voiceSelect.appendChild(option);
        });
        
        setStatus('Ready to synthesize speech.');
        speakButton.disabled = false;
        speakButton.classList.remove('loading');

    } catch (error) {
        setStatus('Failed to load model.');
        console.error('Initialization error:', error);
    }
};

// Function to generate and play speech
const generateSpeech = async () => {
    if (!synthesizer || !speaker_embeddings) {
        setStatus('Synthesizer not ready. Please wait.');
        return;
    }

    const text = textInput.value.trim();
    if (!text) {
        setStatus('Please enter some text.');
        return;
    }

    const speakerId = voiceSelect.value;
    if (!speakerId) {
        setStatus('Please select a voice.');
        return;
    }
    
    setStatus('Synthesizing speech...');
    speakButton.disabled = true;
    speakButton.classList.add('loading');
    audioContainer.innerHTML = '';

    try {
        const speakerEmbedding = speaker_embeddings[speakerId];
        const result = await synthesizer(text, { speaker_embeddings: speakerEmbedding });
        
        const audio = document.createElement('audio');
        audio.src = result.audio;
        audio.controls = true;
        audio.autoplay = true;
        audioContainer.appendChild(audio);

        setStatus('Speech synthesized successfully.');
    } catch (error) {
        setStatus('An error occurred during synthesis.');
        console.error('Synthesis error:', error);
    } finally {
        speakButton.disabled = false;
        speakButton.classList.remove('loading');
    }
};

// Event Listeners
speakButton.addEventListener('click', generateSpeech);

// Initialize the application
initializeSynthesizer();

