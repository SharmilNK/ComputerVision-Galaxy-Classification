import React, { useState, useEffect } from 'react';
import { Client } from '@gradio/client';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [gradCamImage, setGradCamImage] = useState(null);
  const [classificationResult, setClassificationResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gradioClient, setGradioClient] = useState(null);

  // API endpoint - supports both local Gradio and Hugging Face Space
  // For local: http://localhost:7860 or http://YOUR_IP:7860
  // For Hugging Face Space: https://sharmilnk-galaxy-morphology-classification.hf.space
  // Default to Hugging Face Space for public access
  const GRADIO_BASE_URL = import.meta.env.VITE_GRADIO_URL || 'https://sharmilnk-galaxy-morphology-classification.hf.space';

  // Initialize Gradio client
  useEffect(() => {
    const initClient = async () => {
      try {
        const client = await Client.connect(GRADIO_BASE_URL);
        setGradioClient(client);
        const endpoints = Object.keys(client.endpoints || {});
        console.log('Gradio client connected. Available endpoints:', endpoints);
        
        // If no endpoints found, it might be due to API schema bug
        if (endpoints.length === 0) {
          console.warn('No endpoints found. This might be due to Gradio API schema bug.');
          console.warn('The API might still work - try calling predict with index 0');
        }
      } catch (err) {
        console.error('Failed to connect to Gradio:', err);
        setError('Failed to connect to API. Make sure the Gradio server is running.');
      }
    };
    initClient();
  }, [GRADIO_BASE_URL]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setGradCamImage(null);
        setClassificationResult('');
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert base64 to blob
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      
      // For Gradio API, we need to upload the file first or send as base64
      // Option 1: Upload file to Gradio and get file path (for Hugging Face Spaces)
      // Option 2: Send as base64 data URL (works for both local and HF)
      
      // Create a File object from the blob (Gradio client handles File objects well)
      const file = new File([blob], 'galaxy.jpg', { type: blob.type || 'image/jpeg' });

      if (!gradioClient) {
        throw new Error('Gradio client not initialized. Please wait a moment and try again.');
      }

      // Use Gradio client to call the predict endpoint
      // The client expects the endpoint name (without /) and an array of inputs
      // For images, we can pass the File object directly - the client will handle upload
      let result;
      
      // Try with function index 0 first (most reliable when endpoints aren't discovered)
      // This works even when endpoint discovery fails due to schema bug
      try {
        console.log('Attempting API call with index 0...');
        result = await gradioClient.predict(0, [file]);
        console.log('API call successful with index 0');
      } catch (indexError) {
        console.warn('Failed with index 0, trying with endpoint name "predict":', indexError);
        try {
          // Fallback to endpoint name if index doesn't work
          result = await gradioClient.predict('predict', [file]);
          console.log('API call successful with endpoint name');
        } catch (predictError) {
          console.error('Both index and endpoint name failed:', predictError);
          throw new Error(`API call failed: ${predictError.message || predictError}`);
        }
      }
      
      // Gradio client returns either:
      // 1. Direct array: [image, result_text]
      // 2. Object with data property: { type: "data", data: [image, result_text] }
      console.log('API response:', result);
      console.log('Response type:', typeof result, 'Is array:', Array.isArray(result));
      
      // Extract the actual data array from the response
      let resultData;
      if (result && typeof result === 'object') {
        // Check if it's an object with a data property (queue-based response)
        if (result.data && Array.isArray(result.data)) {
          resultData = result.data;
          console.log('Extracted data from response object:', resultData);
        }
        // Check if it's already an array
        else if (Array.isArray(result)) {
          resultData = result;
          console.log('Response is already an array:', resultData);
        }
      } else if (Array.isArray(result)) {
        resultData = result;
      }
      
      if (resultData && Array.isArray(resultData) && resultData.length >= 2) {
        const [gradcamImage, resultText] = resultData;
        console.log('Grad-CAM image type:', typeof gradcamImage, gradcamImage);
        console.log('Result text:', resultText);
        
        // Handle image response (can be base64, file path, or data URL)
        if (gradcamImage) {
          let imageUrl;
          
          if (typeof gradcamImage === 'string') {
            // Check if it's already a data URL
            if (gradcamImage.startsWith('data:')) {
              imageUrl = gradcamImage;
            }
            // Check if it's a file path (from Hugging Face Space)
            else if (gradcamImage.startsWith('/file=') || gradcamImage.startsWith('/')) {
              // Convert file path to full URL
              const fileName = gradcamImage.replace('/file=', '').replace('/', '');
              imageUrl = `${GRADIO_BASE_URL}/file=${fileName}`;
            }
            // Assume it's base64
            else {
              imageUrl = `data:image/png;base64,${gradcamImage}`;
            }
          } else if (gradcamImage && typeof gradcamImage === 'object') {
            // Handle object format (Gradio returns { path: "...", url: "...", ... })
            // For local files, use the url property which points to the Gradio file endpoint
            if (gradcamImage.url) {
              // URL from Gradio is usually already a full URL like "http://localhost:7860/file=..."
              // or might be relative like "/file=..."
              if (gradcamImage.url.startsWith('http://') || gradcamImage.url.startsWith('https://')) {
                imageUrl = gradcamImage.url;
              } else if (gradcamImage.url.startsWith('/file=')) {
                // Relative file URL - prepend base URL
                imageUrl = `${GRADIO_BASE_URL}${gradcamImage.url}`;
              } else if (gradcamImage.url.startsWith('/')) {
                // Other relative URL
                imageUrl = `${GRADIO_BASE_URL}${gradcamImage.url}`;
              } else {
                // Might be just a filename, construct file URL
                imageUrl = `${GRADIO_BASE_URL}/file=${gradcamImage.url}`;
              }
            } else if (gradcamImage.path) {
              // Use path to construct file URL
              // Extract filename from Windows or Unix path
              const path = gradcamImage.path;
              const fileName = path.includes('\\') 
                ? path.split('\\').pop() 
                : path.split('/').pop();
              imageUrl = `${GRADIO_BASE_URL}/file=${fileName}`;
            } else {
              const path = gradcamImage.name || gradcamImage;
              imageUrl = path.startsWith('http') ? path : `${GRADIO_BASE_URL}/file=${path}`;
            }
          }
          
          if (imageUrl) {
            setGradCamImage(imageUrl);
          }
        }
        
        if (resultText) {
          setClassificationResult(resultText);
        }
      } else {
        // Log the actual response to help debug
        console.error('Unexpected response format. Response:', result);
        console.error('Response type:', typeof result);
        if (result) {
          console.error('Response keys:', Object.keys(result));
        }
        throw new Error(`Unexpected API response format. Got: ${JSON.stringify(result).substring(0, 200)}`);
      }
    } catch (err) {
      setError(`Error: ${err.message}. Make sure your API is running and accessible.`);
      console.error('Classification error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Landing Section */}
      <section className="landing-section">
        <div className="landing-image-container">
          <img 
            src="/landing.jpg" 
            alt="Galaxy" 
            className="landing-image"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        </div>
        <div className="landing-text">
          <div className="landing-badge">
            <span className="badge-icon">✨</span>
            <span>AI-Powered Classification</span>
          </div>
          <h1 className="landing-title">
            <span className="title-part-1">Galaxy Morphology</span>
            <span className="title-part-2">AI Classifier</span>
          </h1>
          <p className="landing-subtitle">
            Upload a galaxy image and discover its morphology using state-of-the-art deep learning technology
          </p>
        </div>
      </section>

      {/* Spacing */}
      <div className="section-spacing"></div>

      {/* How Astrophysicists Use This Section */}
      <section className="info-section">
        <div className="info-content">
          <div className="info-card">
            <div className="info-card-content">
              <p>
                Galaxy morphology classification plays a crucial role in understanding dark energy, one of the most profound mysteries in modern cosmology. Dark energy is the mysterious force driving the accelerated expansion of the universe, and its nature remains one of the biggest questions in physics.
              </p>
            </div>
          </div>
          <div className="info-image-container">
            <img 
              src="/astro.jpg" 
              alt="Astrophysics Research" 
              className="info-image"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
            <p className="image-label">Astrophysics Research</p>
          </div>
        </div>
      </section>

      {/* Spacing */}
      <div className="section-spacing"></div>

      {/* Classification Section */}
      <section className="classification-section">
        <h2>Galaxy Morphology Classification</h2>
        <p className="section-description">
          Upload a galaxy image to classify its morphology and visualize the model's attention using Grad-CAM.
        </p>

        <div className="classification-container">
          <div className="input-column">
            <div className="upload-area">
              {selectedImage ? (
                <div className="image-preview">
                  <img src={selectedImage} alt="Uploaded galaxy" />
                  <button 
                    className="remove-image-btn"
                    onClick={() => {
                      setSelectedImage(null);
                      setGradCamImage(null);
                      setClassificationResult('');
                    }}
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <label className="upload-label">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    style={{ display: 'none' }}
                  />
                  <div className="upload-placeholder">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <p>Drop Image Here - or - Click to Upload</p>
                  </div>
                </label>
              )}
            </div>
            <button 
              className="classify-btn"
              onClick={handleClassify}
              disabled={!selectedImage || loading}
            >
              {loading ? (
                <span className="loading-content">
                  <span className="spinner"></span>
                  Classifying...
                </span>
              ) : (
                'Classify Galaxy'
              )}
            </button>
          </div>

          <div className="output-column">
            <div className="output-area">
              <h3>Grad-CAM Visualization</h3>
              {gradCamImage ? (
                <img src={gradCamImage} alt="Grad-CAM visualization" className="output-image" />
              ) : (
                <div className="output-placeholder">
                  <p>Grad-CAM visualization will appear here</p>
                </div>
              )}
            </div>
            <div className="result-area">
              <h3>Classification Result</h3>
              {classificationResult ? (
                <div className="result-text">
                  {classificationResult.split('\n').map((line, idx) => (
                    <p key={idx}>{line}</p>
                  ))}
                </div>
              ) : (
                <div className="result-placeholder">
                  <p>Classification result will appear here</p>
                </div>
              )}
              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Spacing */}
      <div className="section-spacing"></div>

      {/* Science Behind Galaxy Classification Section */}
      <section className="science-section">
        <div className="science-header">
          <h2>The Science Behind Galaxy Classification</h2>
          <p className="science-subtitle">
            Discover how AI-powered galaxy morphology classification is revolutionizing our understanding of the cosmos
          </p>
        </div>
        
        <div className="science-cards">
          <div className="science-card card-purple">
            <div className="card-icon">🔭</div>
            <h3>How Astrophysicists Use This</h3>
            <p>
              Galaxy morphology classification helps researchers analyze data from the Hubble and James Webb Space Telescopes. 
              By identifying elliptical and spiral galaxies, scientists can understand galaxy formation, evolution, and the 
              distribution of matter in the universe.
            </p>
          </div>

          <div className="science-card card-blue">
            <div className="card-icon">🧠</div>
            <h3>Deep Learning Technology</h3>
            <p>
              The model uses convolutional neural networks (ResNet-18) to identify key features like spiral arms, central 
              bulges, and overall structure. This enables processing millions of galaxy images efficiently, accelerating 
              discoveries in cosmology.
            </p>
          </div>

          <div className="science-card card-dark-purple">
            <div className="card-icon">🌌</div>
            <h3>Understanding Dark Energy</h3>
            <p>
              By classifying galaxies and mapping their distribution across cosmic time, astronomers trace the universe's 
              expansion history. Different galaxy types evolve differently, providing clues about dark energy—the mysterious 
              force driving cosmic acceleration.
            </p>
          </div>

          <div className="science-card card-red">
            <div className="card-icon">⭐</div>
            <h3>Future of Space Research</h3>
            <p>
              Automated classification enables analysis of millions of galaxies from surveys like the Vera C. Rubin Observatory's 
              LSST. These datasets will provide unprecedented precision in measuring dark energy and understanding the fate 
              of the universe.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
