import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Initialize global variables
model = None
tokenizer = None
max_len_seq = None

def load_model_artifacts():
    """Load model and tokenizer artifacts for Arabic text generation."""
    global model, tokenizer, max_len_seq
    
    # Paths for files in Kaggle
    model_path = '/kaggle/working/model.h5'
    tokenizer_path = '/kaggle/working/tokenizer.json'
    config_path = '/kaggle/working/config.json'
    
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

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_arabic_text,
    inputs=[
        gr.Textbox(
            label="أدخل النص الأولي الخاص بك",
            placeholder="ابدأ النص هنا...",
            value="الحب هو"
        ),
        gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="عدد الكلمات المراد توليدها"
        )
    ],
    outputs=gr.Textbox(label="النص المُنتج"),
    title="مولد نصوص بالعربية",
    description="""استخدم الذكاء الاصطناعي لتوليد نصوص باللغة العربية! 
                أدخل نصاً أولياً واختر عدد الكلمات التي تريد توليدها. 
                سيتم توليد النص باللغة العربية بناءً على النص الأولي الذي أدخلته.""",
   
    theme=gr.themes.Base()
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
