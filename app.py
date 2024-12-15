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
    
    # Paths for files in the current directory
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

# UI text for both languages
texts = {
    "ar": {
        "title": "مولد نصوص بالعربية",
        "description": """استخدم الذكاء الاصطناعي لتوليد نصوص باللغة العربية! 
                        أدخل نصاً أولياً واختر عدد الكلمات التي تريد توليدها. 
                        سيتم توليد النص باللغة العربية بناءً على النص الأولي الذي أدخلته.""",
        "input_label": "أدخل النص الأولي الخاص بك",
        "output_label": "النص المُنتج",
        "num_words_label": "عدد الكلمات المراد توليدها",
        "placeholder": "ابدأ النص هنا...",
    },
    "en": {
        "title": "Arabic Text Generator",
        "description": """Use AI to generate Arabic text! 
                        Enter a seed text and choose the number of words to generate. 
                        The text will be generated in Arabic based on the seed text you provide.""",
        "input_label": "Enter your seed text",
        "output_label": "Generated Text",
        "num_words_label": "Number of Words to Generate",
        "placeholder": "Start your text here...",
    }
}

def create_interface(language="ar"):
    """Create a Gradio interface based on the selected language."""
    lang_texts = texts[language]
    rtl = "rtl" if language == "ar" else "ltr"
    alignment = "right" if language == "ar" else "left"
    
    return gr.Interface(
        fn=generate_arabic_text,
        inputs=[
            gr.Textbox(
                label=lang_texts["input_label"],
                placeholder=lang_texts["placeholder"],
                value="اه ماشي" if language == "ar" else "Hello",
                elem_id="input-textbox"
            ),
            gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label=lang_texts["num_words_label"]
            )
        ],
        outputs=gr.Textbox(
            label=lang_texts["output_label"],
            elem_id="output-textbox"
        ),
        title=lang_texts["title"],
        description=lang_texts["description"],
        theme=gr.themes.Default(primary_hue="blue"),
        css=f"""
            #input-textbox textarea {{
                text-align: {alignment};
                direction: {rtl};
                font-family: 'Cairo', sans-serif;
                font-size: 18px;
            }}
            
            #output-textbox textarea {{
                text-align: {alignment};
                direction: {rtl};
                font-family: 'Cairo', sans-serif;
                font-size: 18px;
            }}
            
            body {{
                font-family: 'Cairo', sans-serif;
                background: linear-gradient(to right, #4facfe, #00f2fe);
            }}
        """
    )

# Language selection interface
def toggle_language(lang):
    global iface
    iface = create_interface(language=lang)
    iface.launch()

# Default to Arabic interface
iface = create_interface(language="ar")

if __name__ == "__main__":
    iface.launch()
