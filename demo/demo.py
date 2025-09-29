import streamlit as st
import pandas as pd
from PIL import Image
import torch
from plonk import PlonkPipeline
from pathlib import Path
from streamlit_extras.colored_header import colored_header
import plotly.express as px
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Around the World in 80 Timesteps", page_icon="🗺️", layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
# Define checkpoint path
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MODEL_NAMES = {
    "PLONK_YFCC": "nicolas-dufour/PLONK_YFCC",
    "PLONK_OSV_5M": "nicolas-dufour/PLONK_OSV_5M",
    "PLONK_iNaturalist": "nicolas-dufour/PLONK_iNaturalist",
}


@st.cache_resource
def load_model(model_name):
    """Load the model and cache it to prevent reloading"""
    try:
        pipe = PlonkPipeline(model_path=model_name)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


PIPES = {model_name: load_model(MODEL_NAMES[model_name]) for model_name in MODEL_NAMES}


def predict_location(image, model_name, cfg=0.0, num_samples=256):
    with torch.no_grad():
        batch = {"img": [], "emb": []}

        # If image is already a PIL Image, use it directly
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")

        pipe = PIPES[model_name]

        # Get regular predictions
        predicted_gps = pipe(img, batch_size=num_samples, cfg=cfg, num_steps=32)

        # Get single high-confidence prediction
        high_conf_gps = pipe(img, batch_size=1, cfg=2.0, num_steps=32)
        return {
            "lat": predicted_gps[:, 0].astype(float).tolist(),
            "lon": predicted_gps[:, 1].astype(float).tolist(),
            "high_conf_lat": high_conf_gps[0, 0].astype(float),
            "high_conf_lon": high_conf_gps[0, 1].astype(float),
        }


def load_example_images():
    """Load example images from the examples directory"""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        st.error(
            """
            Examples directory not found. Please create the following structure:
            demo/
            └── examples/
                ├── eiffel_tower.jpg
                ├── colosseum.jpg
                ├── taj_mahal.jpg
                ├── statue_liberty.jpg
                └── sydney_opera.jpg
            """
        )
        return {}

    examples = {}
    for img_path in examples_dir.glob("*.jpg"):
        # Use filename without extension as the key
        name = img_path.stem.replace("_", " ").title()
        examples[name] = str(img_path)

    if not examples:
        st.warning("No example images found in the examples directory.")

    return examples


def resize_image_for_display(image, max_size=400):
    """Resize image while maintaining aspect ratio"""
    # Get current size
    width, height = image.size

    # Calculate ratio to maintain aspect ratio
    if width > height:
        if width > max_size:
            ratio = max_size / width
            new_size = (max_size, int(height * ratio))
    else:
        if height > max_size:
            ratio = max_size / height
            new_size = (int(width * ratio), max_size)

    # Only resize if image is larger than max_size
    if width > max_size or height > max_size:
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None


def main():
    # Custom CSS
    st.markdown(
        """
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #FF6B6B;
        }
        .prediction-box {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        /* New styles for image containers */
        .upload-container {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 1rem;
        }
        .examples-container {
            max-height: 200px;
            display: flex;
            gap: 10px;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header with custom styling
    colored_header(
        label="🗺️ Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation",
        description="Upload an image and our model, PLONK, will predict possible locations! In red we will sample one point with guidance scale 2.0 for the best guess. <br> <br> Project page: https://nicolas-dufour.github.io/plonk",
        color_name="red-70",
    )

    # Adjust column ratio to give 2/3 of the space to the map
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Add model selection before the sliders
        model_name = st.selectbox(
            "🤖 Select Model",
            options=MODEL_NAMES.keys(),
            index=0,  # Default to YFCC
            help="Choose which PLONK model variant to use for prediction.",
        )

        # Modify the slider columns to accommodate both controls
        col_slider1, col_slider2 = st.columns([0.5, 0.5])
        with col_slider1:
            cfg_value = st.slider(
                "🎯 Guidance scale",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help="Scale for classifier-free guidance during sampling. A small value makes the model predictions display the diversity of the model, while a large value makes the model predictions more conservative but potentially more accurate.",
            )

        with col_slider2:
            num_samples = st.number_input(
                "🎲 Number of samples",
                min_value=1,
                max_value=5000,
                value=1000,
                step=1,
                help="Number of location predictions to generate. More samples give better coverage but take longer to compute.",
            )

        st.markdown("### 📸 Choose your image")
        tab1, tab2, tab3 = st.tabs(["Upload", "URL", "Examples"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["png", "jpg", "jpeg"],
                help="Supported formats: PNG, JPG, JPEG",
            )

            if uploaded_file is not None:
                st.markdown('<div class="upload-container">', unsafe_allow_html=True)
                original_image = Image.open(uploaded_file)
                display_image = resize_image_for_display(
                    original_image.copy(), max_size=300
                )
                st.image(
                    display_image, caption="Uploaded Image", use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button("🔍 Predict Location", key="predict_upload"):
                    with st.spinner("🌍 Analyzing image and predicting locations..."):
                        predictions = predict_location(
                            original_image,
                            model_name=model_name,
                            cfg=cfg_value,
                            num_samples=num_samples,
                        )
                        st.session_state["predictions"] = predictions

        with tab2:
            url = st.text_input("Enter image URL:", key="image_url")

            if url:
                image = load_image_from_url(url)
                if image:
                    st.markdown(
                        '<div class="upload-container">', unsafe_allow_html=True
                    )
                    display_image = resize_image_for_display(image.copy(), max_size=300)
                    st.image(
                        display_image,
                        caption="Image from URL",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button("🔍 Predict Location", key="predict_url"):
                        with st.spinner(
                            "🌍 Analyzing image and predicting locations..."
                        ):
                            predictions = predict_location(
                                image,
                                model_name=model_name,
                                cfg=cfg_value,
                                num_samples=num_samples,
                            )
                            st.session_state["predictions"] = predictions

        with tab3:
            examples = load_example_images()
            st.markdown('<div class="examples-container">', unsafe_allow_html=True)
            example_cols = st.columns(len(examples))

            for idx, (name, path) in enumerate(examples.items()):
                with example_cols[idx]:
                    original_image = Image.open(path)
                    display_image = resize_image_for_display(
                        original_image.copy(), max_size=150
                    )

                    if st.container().button(
                        "📸",
                        key=f"img_{name}",
                        help=f"Click to predict location for {name}",
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "🌍 Analyzing image and predicting locations..."
                        ):
                            predictions = predict_location(
                                original_image,
                                model_name=model_name,
                                cfg=cfg_value,
                                num_samples=num_samples,
                            )
                            st.session_state["predictions"] = predictions
                            st.rerun()

                    st.image(display_image, caption=name, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🌍 Predicted Locations")

        if "predictions" in st.session_state:
            pred = st.session_state["predictions"]

            # Create DataFrame for all predictions
            df = pd.DataFrame(
                {
                    "lat": pred["lat"],
                    "lon": pred["lon"],
                    "type": ["Sample"] * len(pred["lat"]),
                }
            )

            # Add high-confidence prediction
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "lat": [pred["high_conf_lat"]],
                            "lon": [pred["high_conf_lon"]],
                            "type": ["Best Guess"],
                        }
                    ),
                ]
            )

            # Create a more interactive map using Plotly
            fig = px.scatter_mapbox(
                df,
                lat="lat",
                lon="lon",
                zoom=2,
                opacity=0.6,
                color="type",
                color_discrete_map={"Sample": "blue", "Best Guess": "red"},
                mapbox_style="carto-positron",
            )

            fig.update_traces(selector=dict(name="Best Guess"), marker_size=15)

            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=500,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )

            # Display map in a container
            with st.container():
                st.plotly_chart(fig, use_container_width=True)

            # Display stats in a styled container
            with st.container():
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h4>📊 Prediction Statistics</h4>
                        <p>Number of sampled locations: {len(pred["lat"])}</p>
                        <p>Best guess location: {pred["high_conf_lat"]:.2f}°, {pred["high_conf_lon"]:.2f}°</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Empty state with better styling
            st.markdown(
                """
                <div class="prediction-box" style="text-align: center;">
                    <h4>👆 Upload an image and click 'Predict Location'</h4>
                    <p>The predicted locations will appear here on an interactive map.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
