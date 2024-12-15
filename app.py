import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Initialize global variables
model = None
tokenizer = None
max_len_seq = None

def load_model_artifacts():
    """Load model and tokenizer artifacts for Arabic text generation."""
    global model, tokenizer, max_len_seq
    
    # Paths for files in Kaggle
    model_path = 'model.h5'
    tokenizer_path = 'tokenizer.json'
    config_path = 'config.json'
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load the tokenizer
    with open(tokenizer_path, 'r') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    
    # Load configuration (e.g., max sequence length)
    with open(config_path, 'r') as f:
        config = json.load(f)
        max_len_seq = config['max_len_seq']

def generate_arabic_text(seed_text, num_words):
    """Generate Arabic text from seed text."""
    if model is None:
        load_model_artifacts()
    
    try:
        for _ in range(int(num_words)):
            # Convert the seed text to sequences
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            # Pad the sequences
            token_list = pad_sequences([token_list], maxlen=max_len_seq-1, padding='pre')
            # Predict the next word
            predicted = model.predict(token_list, verbose=0)
            # Get the word with the highest probability
            predicted_word_index = np.argmax(predicted, axis=-1).item()
            predicted_word = tokenizer.index_word.get(predicted_word_index, '')
            # Add the predicted word to the seed text
            seed_text += " " + predicted_word
        
        return seed_text
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Custom theme for the Arabic Text Generator
custom_theme = gr.themes.Default().set(
    button_primary_background="#4CAF50",
    button_primary_color="#FFFFFF",
    textbox_background="#FFFFFF",
    slider_track_color="#2196F3",
    slider_handle_color="#4CAF50"
)

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_arabic_text,
    inputs=[
        gr.Textbox(
            label="أدخل النص الأولي الخاص بك",
            placeholder="ابدأ النص هنا...",
            value="اه ماشي",
            elem_id="input-text"
        ),
        gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="عدد الكلمات المراد توليدها | Num of words",
            elem_id="word-slider"
        )
    ],
    outputs=gr.Textbox(label="النص المُنتج", elem_id="output-text"),
    title="مولد النصوص بالعربية",
    description="""استخدم الذكاء الاصطناعي لتوليد نصوص باللغة العربية! 
                أدخل نصاً أولياً واختر عدد الكلمات التي تريد توليدها.""",
    theme=custom_theme,
    layout="horizontal",  # Arrange input and output side by side
    css="""
        #input-text, #output-text {
            text-align: right;
            direction: rtl;
        }
        #input-text {
            background-color: #f5f5f5;
            border: 2px solid #4CAF50;
        }
        #output-text {
            background-color: #e8f5e9;
            border: 2px solid #2196F3;
        }
        body {
            background: linear-gradient(120deg, #fdfbfb, #ebedee);
            font-family: 'Cairo', sans-serif;
        }
    """
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
