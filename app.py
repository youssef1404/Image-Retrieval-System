import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import os

# Set page title and configuration
st.set_page_config(
    page_title="Image Retrieval System",
    layout="wide"
)

# Constants
IMAGE_DIR = os.path.join(os.getcwd(), 'data', 'images')
DATA_DIR = os.path.join(os.getcwd(), 'artifacts')
NUM_SIMILAR_IMAGES = 5

# Cache data loading to improve performance
@st.cache_data
def load_image_data():
    """Load pre-computed image vectors and filenames."""
    all_vectors = np.load(os.path.join(DATA_DIR, "all_vecs.npy"))
    all_filenames = np.load(os.path.join(DATA_DIR, "all_names.npy"))
    return all_vectors, all_filenames

def display_image(filename, container, caption=None):
    """Display an image in the specified container with optional caption."""
    try:
        img_path = os.path.join(IMAGE_DIR, filename)
        image = Image.open(img_path)
        container.image(image, caption=caption)
        return True
    except Exception as e:
        container.error(f"Error loading image: {e}")
        return False

def get_image_index(filename, all_filenames):
    """Find the index of an image filename in the array of all filenames."""
    # Find where the filename matches in the array and return the index
    matches = np.where(all_filenames == filename)[0]
    if len(matches) > 0:
        return matches[0]
    return None

def find_similar_images(target_index, vectors, filenames, num_similar=NUM_SIMILAR_IMAGES):
    """Find the most similar images to the target image."""
    target_vector = vectors[target_index].reshape(1, -1)
    
    # Calculate distances between target vector and all other vectors
    distances = cdist(target_vector, vectors).squeeze()
    
    # Get indices of most similar images (excluding the target image itself)
    # Sort distances and take indices from position 1 (skipping the target image at position 0)
    similar_indices = distances.argsort()[1:num_similar+1]
    similar_distances = distances[similar_indices]
    
    return similar_indices, similar_distances

def normalize_similarity(distance, max_distance=None):
    """Convert distance to similarity score (0-100%)."""
    if max_distance is None:
        # If no max_distance provided, use a reasonable default
        max_distance = 2.0
    
    # Ensure the similarity is between 0 and 1
    similarity = max(0, min(1, 1 - (distance / max_distance)))
    return similarity

def main():
    st.title("Image Retrieval System")
    
    # Load data
    with st.spinner("Loading image data..."):
        vectors, filenames = load_image_data()
    
    # Initialize session state if not already done
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
        st.session_state.current_index = None
        st.session_state.similar_indices = None
        st.session_state.similar_distances = None
    
    # Layout
    main_col1, main_col2, main_col3 = st.columns([1, 2, 1])
    
    # Action buttons
    with st.container():
        button_col1, button_col2 = st.columns(2)
        random_button = button_col1.button("Show Random Image", use_container_width=True)
        similar_button = button_col2.button("Find Similar Images", 
                                            use_container_width=True, 
                                            disabled=st.session_state.current_image is None)
        
    # Handle random image selection
    if random_button:
        # Select random image
        random_index = np.random.randint(len(filenames))
        random_filename = filenames[random_index]
        st.session_state.current_image = random_filename
        st.session_state.current_index = random_index
        # Reset similar images when a new main image is selected
        st.session_state.similar_indices = None
        st.session_state.similar_distances = None
    
    # Display current image
    with main_col2:
        if st.session_state.current_image:
            st.subheader("Current Image")
            display_image(st.session_state.current_image, st.container())
            
            # Debug info (can be removed in production)
            with st.expander("Debug Info"):
                st.write(f"Current image: {st.session_state.current_image}")
                st.write(f"Index: {st.session_state.current_index}")
    
    # Find and display similar images
    if similar_button and st.session_state.current_image and st.session_state.current_index is not None:
        # Find similar images
        similar_indices, similar_distances = find_similar_images(
            st.session_state.current_index, vectors, filenames
        )
        
        # Store in session state
        st.session_state.similar_indices = similar_indices
        st.session_state.similar_distances = similar_distances
    
    # Display similar images if available
    if st.session_state.similar_indices is not None:
        st.subheader("Similar Images")
        
        # Calculate the maximum distance for normalization
        # This helps spread out the similarity percentages
        max_distance = max(st.session_state.similar_distances) * 1.2  # Add 20% buffer
        
        # Display similar images in a grid
        similar_cols = st.columns(NUM_SIMILAR_IMAGES)
        
        for i, (idx, distance) in enumerate(zip(st.session_state.similar_indices, 
                                            st.session_state.similar_distances)):
            similar_filename = filenames[idx]
            
            # Calculate normalized similarity score (0-100%)
            similarity = normalize_similarity(distance, max_distance)
            similarity_percentage = f"{similarity:.0%} similar"
            
            display_image(similar_filename, similar_cols[i], caption=similarity_percentage)

if __name__ == "__main__":
    main()