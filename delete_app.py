import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# Workaround for Gradio 6.0.1 API schema bug
# Fix: Cannot parse schema True - boolean values in schema cause errors
try:
    import gradio_client.utils as client_utils
    
    # Patch _json_schema_to_python_type to handle boolean schemas
    if hasattr(client_utils, '_json_schema_to_python_type'):
        original_json_schema_to_python_type = client_utils._json_schema_to_python_type
        
        def patched_json_schema_to_python_type(schema, _depth=0):
            if isinstance(schema, bool):
                return "bool"
            return original_json_schema_to_python_type(schema, _depth)
        
        client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    
    # Also patch get_type as fallback
    if hasattr(client_utils, 'get_type'):
        original_get_type = client_utils.get_type
        
        def patched_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            return original_get_type(schema)
        
        client_utils.get_type = patched_get_type
except Exception as e:
    print(f"Warning: Could not patch Gradio client utils: {e}")
    pass  # If patching fails, continue anyway

# Model configuration
MODEL_PATH = "robust_galaxy_model.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["Elliptical", "Spiral"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Load model
def get_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything (match training)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_model():
    model = get_model(NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
            print(f"Model loaded successfully from {MODEL_PATH}")
            print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
            
            # Test with a dummy input to verify model works
            test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                test_output = model(test_input)
                print(f"Model test output shape: {test_output.shape}")
                print(f"Model test output: {test_output[0].cpu().numpy()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            print("Using untrained model")
    else:
        print(f"Model file not found at {MODEL_PATH}. Using untrained model.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
    model.to(DEVICE)
    model.eval()
    return model

# Load model - handle errors gracefully
model = None
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    # Create a dummy model as fallback
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.eval()
    print("Using untrained model as fallback")

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        try:
            # Forward pass
            model_output = self.model(input_image)
            
            if target_class is None:
                target_class = model_output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            class_score = model_output[0, target_class]
            class_score.backward(retain_graph=False)
            
            if self.gradients is None or self.activations is None:
                return np.zeros((7, 7))  # Default size for ResNet layer4
            
            gradients = self.gradients[0]
            activations = self.activations[0]
            
            # Global average pooling of gradients
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activations).sum(dim=0)
            
            # Apply ReLU and normalize
            cam = F.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam.detach().cpu().numpy()
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
            self.gradients = None
            self.activations = None

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    output = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return output

def predict_galaxy(image):
    """Predict galaxy morphology and generate Grad-CAM (probabilities only)"""
    if image is None:
        return None, "Please upload an image."

    if model is None:
        return None, "Error: Model not loaded. Please check the logs."

    try:
        model.eval()

        # Convert image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True

        # Forward pass (gradients enabled for Grad-CAM)
        with torch.set_grad_enabled(True):
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)

            raw_probs = probs[0].detach().cpu().numpy()
            pred_class = int(np.argmax(raw_probs))

        # Generate Grad-CAM
        try:
            gradcam = GradCAM(model, model.layer4)
            cam = gradcam.generate_cam(img_tensor, pred_class)
        except Exception:
            cam = None

        # Prepare image for overlay
        img_np = np.array(image)
        img_resized = cv2.resize(img_np, (224, 224))

        if cam is not None:
            overlay = overlay_heatmap(img_resized, cam)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay)
        else:
            overlay_pil = image.resize((224, 224))

        pred_prob = raw_probs[pred_class]
        
        result_text = (
            f"Predicted Class: {CLASS_NAMES[pred_class]}\n"
            f"Probability: {pred_prob:.2%}"
)  

        return overlay_pil, result_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during prediction: {str(e)}"

# Custom CSS for black background and white text
custom_css = """
    .gradio-container {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    body {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    /* Remove white backgrounds from all containers */
    div:not(.gr-textbox):not(.gr-button):not(.gr-image) {
        background-color: #000000 !important;
    }
    .gradio-container, .gradio-container > * {
        background-color: #000000 !important;
    }
    .gradio-container * {
        color: #ffffff !important;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #ffffff !important;
    }
    .gr-markdown, .gr-markdown * {
        color: #ffffff !important;
    }
    .gr-button {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    .gr-button:hover {
        background-color: #555555 !important;
    }
    .gr-textbox, .gr-textbox input, .gr-textbox textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    /* Fix white boxes in output areas */
    .gr-box, .panel, .output-panel, .gr-component {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    .gr-column, .gr-row {
        background-color: #000000 !important;
    }
    /* Output image containers */
    .output-image, .gr-image-output, .image-output {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    /* Result textbox specifically */
    .result-text, [data-testid*="textbox"], .gr-textbox[data-testid*="textbox"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    input[type="text"], textarea, input[type="text"]:focus, textarea:focus {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    .gr-textbox input::-webkit-input-placeholder,
    .gr-textbox textarea::-webkit-input-placeholder {
        color: #888888 !important;
    }
    .gr-textbox input::placeholder,
    .gr-textbox textarea::placeholder {
        color: #888888 !important;
    }
    .output-text, .result-text, .classification-result {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    /* Chrome-specific fixes for textboxes */
    input, textarea, select {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    .gr-textbox-wrapper, .gr-textbox-container {
        background-color: #1a1a1a !important;
    }
    /* Fix for Chrome autofill */
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus,
    textarea:-webkit-autofill,
    textarea:-webkit-autofill:hover,
    textarea:-webkit-autofill:focus {
        -webkit-text-fill-color: #ffffff !important;
        -webkit-box-shadow: 0 0 0px 1000px #1a1a1a inset !important;
        box-shadow: 0 0 0px 1000px #1a1a1a inset !important;
        background-color: #1a1a1a !important;
    }
    /* Fix all white boxes - comprehensive approach */
    div[class*="box"], div[class*="panel"], div[class*="container"] {
        background-color: #000000 !important;
    }
    .gr-box, .panel, .output-panel, .gr-component, .gr-form {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    .gr-column, .gr-row {
        background-color: #000000 !important;
    }
    /* Output image containers */
    .output-image, .gr-image-output, .image-output, div[data-testid*="image"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
    }
    /* All textboxes including output ones */
    .gr-textbox, .gr-textbox input, .gr-textbox textarea, div[data-testid*="textbox"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    .gr-image {
        background-color: #000000 !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gr-image img {
        border: none !important;
        box-shadow: none !important;
        background-color: #000000 !important;
    }
    .gr-image-container, .image-container, .image-wrapper {
        border: none !important;
        background-color: #000000 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gr-image .toolbar, .gr-image .image-controls {
        display: none !important;
    }
    .gr-image label, .gr-image .label-wrap {
        display: none !important;
    }
    .gr-box {
        border: none !important;
        background-color: #000000 !important;
    }
    .panel, .panel-header {
        background-color: #000000 !important;
        border: none !important;
    }
"""

# Create Gradio interface
# Note: In Gradio 6.0+, CSS parameter was removed from Blocks
# CSS styling removed for compatibility - React UI handles its own styling
with gr.Blocks() as demo:
    # Landing Section
    with gr.Column():
        landing_img = gr.Image(value="landing.jpg", height=500, show_label=False, container=False)
        landing_text = gr.Markdown("""
        <div style="text-align: center; padding: 30px; color: white; background-color: #000000; width: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <h1 style="font-size: 96px; font-weight: bold; margin: 0 auto 30px auto; text-align: center; width: 100%;">Galaxy Morphology AI</h1>
            <p style="font-size: 56px; font-weight: normal; margin: 0 auto; text-align: center; width: 100%;">Classify galaxies with state-of-the-art deep learning</p>
        </div>
        """)
    
    # Spacing between sections
    gr.Markdown("<div style='height: 60px;'></div>")
    
    # How Astrophysicists Use This Section
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            # How Astrophysicists Use This
            
            Galaxy morphology classification is a fundamental tool in modern astrophysics. 
            By automatically identifying whether a galaxy is elliptical or spiral, researchers 
            can analyze large datasets from telescopes like the Hubble Space Telescope and 
            the James Webb Space Telescope. This classification helps understand galaxy 
            formation, evolution, and the distribution of matter in the universe.
            
            The deep learning model uses convolutional neural networks to identify key 
            features in galaxy images, such as spiral arms, central bulges, and overall 
            structure. This automated classification enables astronomers to process millions 
            of galaxy images efficiently, accelerating discoveries in cosmology and 
            extragalactic astronomy.
            """)
        with gr.Column(scale=1):
            astro_img = gr.Image(value="astro.jpg", show_label=False, container=False, height=400)
            gr.Markdown("<p style='text-align: center; color: white; margin-top: 10px;'>Astrophysics Research</p>")
    
    # Spacing between sections
    gr.Markdown("<div style='height: 60px;'></div>")
    
    # Classification Section
    gr.Markdown("# Galaxy Morphology Classification")
    gr.Markdown("Upload a galaxy image to classify its morphology and visualize the model's attention using Grad-CAM.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Galaxy Image")
            classify_btn = gr.Button("Classify Galaxy")
        
        with gr.Column():
            output_image = gr.Image(label="Grad-CAM Visualization")
            result_text = gr.Textbox(label="Classification Result")
    
    # Register the classification function
    # Enable API for React frontend integration
    classify_btn.click(
        fn=predict_galaxy,
        inputs=[input_image],
        outputs=[output_image, result_text],
        api_name="predict"
    )
    
    # Spacing between sections
    gr.Markdown("<div style='height: 60px;'></div>")
    
    # Dark Energy Section
    gr.Markdown("""
    # Understanding Dark Energy Through Galaxy Morphology
    
    Galaxy morphology classification plays a crucial role in understanding dark energy, 
    one of the most profound mysteries in modern cosmology. Dark energy is the 
    mysterious force driving the accelerated expansion of the universe, and its nature 
    remains one of the biggest questions in physics.
    
    By classifying large numbers of galaxies and mapping their distribution across 
    cosmic time, astronomers can trace the expansion history of the universe. 
    Different galaxy types (elliptical vs spiral) form and evolve differently, and 
    their relative abundances at different redshifts provide clues about the universe's 
    evolution. The distribution and clustering of these galaxies help measure the 
    large-scale structure of the universe, which is directly influenced by dark energy.
    
    Automated classification systems like this one enable the analysis of millions of 
    galaxies from current and future surveys, such as the Vera C. Rubin Observatory's 
    Legacy Survey of Space and Time (LSST). These massive datasets will provide 
    unprecedented precision in measuring dark energy's properties and understanding 
    its role in the fate of the universe.
    """)

# Launch the demo
# For Hugging Face Spaces, Gradio will automatically detect and launch the demo
# API is enabled by default in Gradio 6.0+
if __name__ == "__main__":
    print("Starting Gradio server...")
    print("API will be available at: http://localhost:7860/gradio_api/predict")
    # Launch with API enabled (default in Gradio 6.0+)
    # Enable queue for API calls
    demo.queue()
    # Allow network access: server_name="0.0.0.0" makes it accessible on local network
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
