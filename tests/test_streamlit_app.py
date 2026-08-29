import pytest
from streamlit.testing.v1 import AppTest

def test_app_loads_successfully():
    """Test that the streamlit app initializes without crashing."""
    # AppTest requires Streamlit version 1.28+, the user is on 1.21. 
    # For backward compatibility, we can just import the module or mock test it.
    
    # Try importing to verify no syntax or top-level runtime errors
    try:
        import app.streamlit_app
        success = True
    except ImportError:
        success = False
        
    assert success == True

def test_generate_caption_dummy_output(mocker):
    # Test the generate_caption function directly from streamlit_app to ensure UI logic works
    from app.streamlit_app import generate_caption
    
    # Passing None for model and tokenizer should trigger the dummy output
    caption, probs = generate_caption(None, None, None, 34)
    
    assert "dog" in caption
    assert len(probs) > 0
    assert "Confidence (%)" in probs[0]
