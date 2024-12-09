import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import Counter
import re
import string
from pyvi import ViTokenizer
from wordcloud import WordCloud, ImageColorGenerator
import random
from streamlit_image_select import image_select
import glob



def single_color_func(color):
    def color_func(*args, **kwargs):
        return color
    return color_func

def multiple_color_func(color_list):
    def color_func(*args, **kwargs):
        return random.choice(color_list)
    return color_func
def load_excel_data(excel_file):
    """
    Loads the Excel file and validates required columns.

    Parameters:
        excel_file (UploadedFile): The uploaded Excel file.

    Returns:
        pd.DataFrame: Loaded DataFrame if successful.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {'Title', 'Content', 'ParentId', 'Sentiment', 'Channel'}
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {e}")
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    return df

def load_stopwords(stopwords_file_path):
    """
    Loads Vietnamese stopwords from a .txt file.

    Parameters:
        stopwords_file_path (str): The path to the stopwords file.

    Returns:
        list: List of stopwords.
    """
    try:
        with open(stopwords_file_path, "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
    except Exception as e:
        raise ValueError(f"Error loading stopwords file: {e}")
    return stopwords

def load_mask_image(mask_image_file):
    """
    Loads and processes the mask image.

    Parameters:
        mask_image_file: 
            - Could be a Streamlit UploadedFile object (file-like)
            - Or a PIL Image object if you've already opened it elsewhere.

    Returns:
        np.ndarray: Numpy array of the mask image.
    """
    try:
        # Check if we already have a PIL Image
        if isinstance(mask_image_file, Image.Image):
            mask_image = mask_image_file
        else:
            # Otherwise, assume it's a file-like object and open it
            mask_image = Image.open(mask_image_file)

        if mask_image.mode != "RGB":
            mask_image = mask_image.convert("RGB")
        mask_array = np.array(mask_image)

    except Exception as e:
        raise ValueError(f"Error loading mask image: {e}")
    
    return mask_array

def filter_data(df, selected_sentiments, selected_channels):
    """
    Filters the DataFrame based on selected sentiments and channels.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        selected_sentiments (list): List of selected sentiments.
        selected_channels (list): List of selected channels.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df['Sentiment'].isin(selected_sentiments)) &
        (df['Channel'].isin(selected_channels))
    ]
    return filtered_df



def tokenize_grouped(df, stopwords, process='content'):
    """
    Tokenizes either the 'Title' or 'Content' column after grouping by 'ParentId'.
    """
    # Validate the 'process' parameter
    if process.lower() not in ['title', 'content']:
        raise ValueError("Parameter 'process' must be either 'title' or 'content'.")

    # Select the column to process
    column_to_process = 'Title' if process.lower() == 'title' else 'Content'

    # Fill NaN values and ensure the column is of string type
    df[column_to_process] = df[column_to_process].fillna('').astype(str)

    # Group by 'ParentId' and concatenate the selected column's text
    grouped_df = df.groupby('ParentId')[column_to_process].apply(' '.join).reset_index()

    # Tokenize the combined text and remove stopwords
    grouped_df['Tokenized'] = grouped_df[column_to_process].apply(
        lambda x: ' '.join([
            word for word in ViTokenizer.tokenize(x).split() 
            if word.lower() not in stopwords  # Ensure case-insensitive stopword removal
        ])
    )

    # Optionally, drop the original concatenated column
    grouped_df = grouped_df.drop(columns=[column_to_process])

    return grouped_df


def generate_frequency_or_tfidf_table(df, use_tfidf=False, min_word_length=0, exclude_keywords=[]):
    """
    Generates a frequency or TF-IDF table from the tokenized text.

    Parameters:
        df (pd.DataFrame): The tokenized DataFrame with 'ParentId' and 'Tokenized' columns.
        use_tfidf (bool): Whether to calculate TF-IDF instead of word frequency.

    Returns:
        pd.DataFrame: Table sorted by frequency or TF-IDF scores.
    """
    if 'Tokenized' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Tokenized' column.")

    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # TF-IDF calculation
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['Tokenized'])
        tfidf_scores = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=vectorizer.get_feature_names_out()
        )
        # Sum TF-IDF scores across all documents
        tfidf_sum = tfidf_scores.sum(axis=0).reset_index()
        tfidf_sum.columns = ['Word', 'TF-IDF_Score']
        # Ensure 'TF-IDF_Score' is float
        tfidf_sum['TF-IDF_Score'] = pd.to_numeric(tfidf_sum['TF-IDF_Score'], errors='coerce')
        # Drop any rows with NaN in 'TF-IDF_Score'
        tfidf_sum = tfidf_sum.dropna(subset=['TF-IDF_Score'])

        # Remove punctuation and non-alphanumeric characters
        tfidf_sum['Word'] = tfidf_sum['Word'].apply(lambda w: re.sub(r'[^\w\s]', '', w))

        # Keep only words that match the given pattern (alphanumeric and whitespace)
        tfidf_sum = tfidf_sum[tfidf_sum['Word'].str.match(r'^[\w\s]+$')]

        # Filter words by minimum length
        tfidf_sum = tfidf_sum[tfidf_sum['Word'].str.len() >= min_word_length]

        # Sort by TF-IDF score in descending order
        frequency_table = tfidf_sum.sort_values(by='TF-IDF_Score', ascending=False).reset_index(drop=True)
    else:
        # Frequency-based calculation
        # Combine all tokenized texts into a single string
        combined_text = ' '.join(df['Tokenized'].astype(str))
        
        # Text Cleaning
        words = combined_text.split()
        # Remove punctuation and non-alphanumeric characters
        words = [
            re.sub(r'[^\w\s]', '', word) for word in words 
            if re.match(r'^[\w\s]+$', word)
        ]
        # Remove empty strings resulting from cleaning
        words = [word for word in words if len(word) >= min_word_length]
        
        # Count word frequencies
        word_counts = Counter(words)
        # Replace underscores with spaces if any (optional based on your tokenizer)
        cleaned_word_counts = {str(word).replace('_', ' '): count for word, count in word_counts.items()}
        
        # Create DataFrame from word counts
        frequency_table = pd.DataFrame(
            list(cleaned_word_counts.items()), 
            columns=['Word', 'Frequency']
        )
        # Ensure 'Frequency' is integer
        frequency_table['Frequency'] = pd.to_numeric(frequency_table['Frequency'], errors='coerce')
        # Drop any rows with NaN in 'Frequency'
        frequency_table = frequency_table.dropna(subset=['Frequency'])
        # Convert 'Frequency' to integer type
        frequency_table['Frequency'] = frequency_table['Frequency'].astype(int)
        # Sort by frequency in descending order
        frequency_table = frequency_table.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    # Optionally, exclude specific keywords
    if exclude_keywords:
        # Trim and split into a list
        exclude_keywords = [word.strip().lower() for word in exclude_keywords.split(",") if word.strip()]

    # Now exclude_keywords is guaranteed to be a list (if not empty)
    if exclude_keywords:
        frequency_table = frequency_table[~frequency_table['Word'].str.lower().isin(exclude_keywords)]
    return frequency_table

def generate_wordcloud(frequency_table, mask_array, customization_options):
    """
    Generates a word cloud based on the frequency table and customization options.

    Parameters:
        frequency_table (pd.DataFrame): Frequency table of words.
        mask_array (np.ndarray): Mask image array.
        customization_options (dict): Dictionary of customization parameters.

    Returns:
        PIL.Image: Generated word cloud image.
    """
    clean_word_dict = {}
    # Assuming frequency_table is a Pandas DataFrame
    frequency_table["Word"] = frequency_table["Word"].astype(str)
    for index, row in frequency_table.iterrows():
        try:
            # Access column values using iloc by index
            word = str(row.iloc[0])  # Column at index 0
            freq = int(row.iloc[1])  # Column at index 1
            clean_word_dict[word] = freq
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # Raise an error if the dictionary is empty
    if not clean_word_dict:
        raise ValueError("The word dictionary is empty after cleaning. Check the input data.")

    wordcloud = WordCloud(
        width=customization_options['width'],
        height=customization_options['height'],
        background_color=customization_options['background_color'],
        mode=customization_options['mode'],
        colormap=customization_options['color_map'],
        mask=mask_array,
        contour_width=customization_options['contour_width'] if customization_options['add_contour'] else 0,
        contour_color=customization_options['contour_color'] if customization_options['add_contour'] else None,
        random_state=42,
        max_words=customization_options['max_words'],
        color_func=customization_options['color_function']
    ).generate_from_frequencies(clean_word_dict)
    
    return wordcloud.to_image()

import streamlit as st
import os
import glob
from PIL import Image
from io import BytesIO
import pandas as pd

# Import your custom functions here
# from your_module import load_excel_data, load_stopwords, filter_data, tokenize_grouped, generate_frequency_or_tfidf_table, generate_wordcloud, image_select, single_color_func, multiple_color_func, load_mask_image

def main():
    st.sidebar.title("Word Cloud Generator")
    
    # Sidebar: Upload Files, Filters, and Processing Options
    st.sidebar.header("Upload Files")
    
    script_dir = os.path.dirname(__file__)
    resources_dir = os.path.join(script_dir, "resources")

    default_excel_file_path = os.path.join(resources_dir, "test.xlsx")
    excel_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    stopwords_file = os.path.join(resources_dir, "vietnamese-stopwords.txt")
    if excel_file is not None:
        # Use the uploaded Excel file
        excel_file_path = excel_file
        using_default = False
    else:
        # Fall back to the default Excel file
        if os.path.exists(default_excel_file_path):
            excel_file_path = default_excel_file_path
            using_default = True
        else:
            st.sidebar.warning("No Excel file uploaded and no default Excel file found. Using no file.")
            excel_file_path = None
            using_default = False

    if not (excel_file_path and os.path.exists(stopwords_file)):
        st.sidebar.info("Please upload all required files to proceed.")
        # Optionally, you can exit early or proceed with defaults if available
        if using_default:
            pass  # Continue if default Excel is available
        else:
            return  # Exit early if no valid Excel file is present

    # Load data
    try:
        df = load_excel_data(excel_file_path)
        stopwords = load_stopwords(stopwords_file)
    except ValueError as ve:
        st.sidebar.error(str(ve))
        return

    # Sidebar Filters
    st.sidebar.header("Filters")
    df['Sentiment'] = df['Sentiment'].fillna('NaN')
    df['Channel'] = df['Channel'].fillna('NaN')
    sentiment_options = df['Sentiment'].dropna().unique().tolist()
    selected_sentiment = st.sidebar.multiselect(
        "Select Sentiment",
        options=sentiment_options,
        default=sentiment_options
    )
    
    channel_options = df['Channel'].dropna().unique().tolist()
    selected_channel = st.sidebar.multiselect(
        "Select Channel",
        options=channel_options,
        default=channel_options
    )
    
    st.sidebar.header("TF-IDF Options")
    use_tfidf = st.sidebar.checkbox("Use TF-IDF", value=False, help="Use TF-IDF instead of frequency table (longer time)")
    
    # Sidebar Processing Options
    st.sidebar.header("Processing Options")
    processing_options = st.sidebar.selectbox(
        options=["Content", "Title"],
        label="Select Columns to Process"
    )
    
    # Apply Filters
    filtered_df = filter_data(df, selected_sentiment, selected_channel)
    
    # Determine which columns to process based on user selection
    process = processing_options
    
    # Main Area: Word Cloud Customization
    with st.container(border=True):
        st.header("Word Cloud Customization")
        col1, col2, col3 = st.columns([2,1,1])
        
        # 1. Color Map Selection
        with col1:
            color_map = st.selectbox(
                "Select Color Map",
                options=[
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
                    'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                    'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
                    'PuBuGn', 'BuGn', 'YlGn'
                ],
                index=0,
                help="Choose a color scheme for the word cloud."
            )
            quality_options = {
                "High Quality": {"width": 1600, "height": 800},
                "Medium Quality": {"width": 800, "height": 400},
                "Low Quality": {"width": 400, "height": 200}
            }
            url = "https://kristendavis27.medium.com/wordcloud-style-guide-2f348a03a7f8"
            url2 = "https://www.croxyproxy.com/"
            st.write("Check out this [link](%s) to see all color maps." % url)
            st.markdown("If you can't access the link, paste it to [this site](%s)." % url2)            
        
        with col2:
            # 2. Image Quality Selection
            quality = st.radio(
                "Select Image Quality",
                options=["High Quality", "Medium Quality", "Low Quality"],
                index=1,  # Default to Medium Quality
                help="Choose the quality of the word cloud image, which determines its dimensions."
            )
        
            width = quality_options[quality]["width"]
            height = quality_options[quality]["height"]
        
        with col3:
            # 4. Maximum Number of Words
            max_words = st.number_input(
                "Maximum Number of Words",
                value=100,
                min_value=50,
                help="Set the maximum number of words to include in the word cloud."
            )

            # 3. Minimum Word Length
            min_word_length = st.number_input(
                "Minimum Word Length",
                value=2,
                min_value=0,
                help="Set the minimum length of words to include in the word cloud."
            )
        
        with st.expander("Choose a Mask Image"):
            mask_dir = os.path.join(resources_dir, "mask")
                                    
            # Get all image file paths in the mask directory
            image_paths = glob.glob(os.path.join(mask_dir, "*"))
        

            # Helper function to extract the first number in the file name
            def extract_number(file_name):
                match = re.search(r"\d+", file_name)
                return int(match.group()) if match else None

            # Sort the image paths based on the first number in the file name
            image_paths.sort(key=lambda x: (extract_number(os.path.basename(x)) is None, extract_number(os.path.basename(x))))

            # Process images to handle RGBA mode
            images = []
            captions = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    # Convert RGBA to RGB if needed
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    images.append(img)
                    captions.append(os.path.basename(img_path))  # Use the file name as the caption
                except Exception as e:
                    print(f"Could not load image {img_path}: {e}")
        
            with st.container():
                # Display the image selection widget
                selected_image = image_select(
                    label="Select an image",
                    images=images,
                    captions=captions
                )

                # File uploader widget
                uploaded_file = st.file_uploader("Or Upload Your Own (e.g., PNG, JPG)", type=["png", "jpg", "jpeg"])

                # Determine the image to use
                if uploaded_file is not None:
                    mask_image_file = uploaded_file  # Use uploaded file
                    st.info("Using uploaded image.")
                else:
                    mask_image_file = selected_image  # Use selected image from the list

                try:
                    # Load the mask image using the determined source
                    mask_array = load_mask_image(mask_image_file)
                except ValueError as ve:
                    st.sidebar.error(f"Error: {str(ve)}")
                    mask_array = None
        
        with st.expander("Advanced Options"):
            with st.container():    
                exclude_keywords = st.text_input("Enter keywords to exclude (comma-separated)", value="", placeholder='Example1, Example2')

            with st.container():        
                # 5. Contour Options
                add_contour = st.checkbox(
                    "Add Contour",
                    value=False,
                    help="Add contour lines to the word cloud."
                )
                col4, col5 = st.columns([1,3])

                if add_contour:
                    with col4:
                        contour_color = st.color_picker(
                            "Contour Color",
                            value="#00FFAA",
                            help="Choose the color of the contour lines."
                        )
                        background_color = 'white'
                        mode = "RGB"
                    with col5:
                        contour_width = st.slider(
                            "Contour Width",
                            min_value=0,
                            max_value=10,
                            value=1,
                            step=1,
                            help="Set the width of the contour lines."
                        )
                else:
                    contour_color = "#00FFAA"
                    contour_width = 0
                    background_color = "rgba(255, 255, 255, 0)"
                    mode = "RGBA"

            with st.container():
                # 6. Color function
                color_function_selection = st.selectbox(
                    "Color Function",
                    options=["Using Color Map", "Using Masked Image Color", "Customize"],
                    index=0,
                    help="Choose the color function for the word cloud (Default = Color Map)"
                )

                if 'custom_colors' not in st.session_state:
                    st.session_state.custom_colors = ["#00FFAA"]  # Default color

                colors = None

                if color_function_selection == "Customize":
                    st.write("Customize your colors:")
                    if st.button("Add Color"):
                        st.session_state.custom_colors.append("#00FFAA")  # Add a new default color picker
                    
                    updated_colors = []
                    # Create columns based on the number of colors
                    cols = st.columns(len(st.session_state.custom_colors))

                    for i, clr in enumerate(st.session_state.custom_colors):
                        with cols[i]:
                            updated_color = st.color_picker(f"Color {i+1}", value=clr)
                            updated_colors.append(updated_color)
                    # Update session state to store the newly chosen colors
                    st.session_state.custom_colors = updated_colors
                    # Otherwise, return a multiple color function
                    if len(st.session_state.custom_colors) == 1:
                        colors = single_color_func(st.session_state.custom_colors[0])
                    else:
                        colors = multiple_color_func(st.session_state.custom_colors)

                elif color_function_selection == "Using Masked Image Color":
                    if mask_array is not None:
                        image_colors = ImageColorGenerator(mask_array)  # Use the subsampled mask image
                        colors = image_colors                        
                        
                    else:
                        st.warning("No mask array provided, defaulting to None.")
                        colors = None
                elif color_function_selection == "Using Color Map":
                    # When using a colormap, set colors to None
                    colors = None

        # Compile customization options into a dictionary
        customization_options = {
            'color_map': color_map,
            'max_words': max_words,
            'width': width,
            'height': height,
            'add_contour': add_contour,
            'contour_color': contour_color,
            'contour_width': contour_width,
            'min_word_length': min_word_length, 
            'color_function': colors,
            'background_color': background_color,
            'mode': mode
        }
        
    # Initialize session state for frequency table and word cloud if not already set
    if 'frequency_table' not in st.session_state:
        st.session_state['frequency_table'] = pd.DataFrame()
    if 'wordcloud' not in st.session_state:
        st.session_state['wordcloud'] = None
    
    # Determine if we should automatically generate the word cloud
    auto_generate = using_default and not st.session_state['frequency_table'].empty == False

    # Add a button to generate the frequency table and word cloud 
    generate_button = st.button("Generate Word Cloud")

    # Automatically generate if using default and not already generated
    if auto_generate:
        generate_button = True  # Simulate button press

    if generate_button:
        if not process:
            st.error("Please select at least one column to process.")
        else:
            try:
                processed_df = tokenize_grouped(filtered_df, stopwords, process=process)
                frequency_table = generate_frequency_or_tfidf_table(processed_df, use_tfidf, min_word_length, exclude_keywords=exclude_keywords)
                if frequency_table.empty:
                    st.error("The frequency table is empty. Please check your data and stopwords.")
                else:
                    st.session_state['frequency_table'] = frequency_table
                    st.session_state['wordcloud'] = generate_wordcloud(frequency_table, mask_array, customization_options)
                    st.success("Word cloud generated successfully!")
                    
                    # Display the initial word cloud immediately
                    st.image(st.session_state['wordcloud'], use_container_width=True)
                    
                    # Provide a download button right after the first generation
                    buf = BytesIO()
                    st.session_state['wordcloud'].save(buf, format="PNG", optimize=True, quality=100)
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Word Cloud as PNG",
                        data=byte_im,
                        file_name='wordcloud.png',
                        mime='image/png',
                    )

            except Exception as e:
                st.error(f"Error generating word cloud: {e}")

    # Collapsible Frequency Table with Editable Data Editor
    with st.expander("Edit Frequency Table"):
        if st.session_state['frequency_table'].empty:
            st.write("Frequency table will appear here after generating.")
        else:
            try:
                edited_freq_table = st.data_editor(
                    st.session_state['frequency_table'],
                    num_rows="dynamic",
                    key="freq_table"
                )
                st.session_state['frequency_table'] = edited_freq_table  # Update session state with edits
                st.session_state['wordcloud'] = None  # Reset wordcloud when frequency table is edited
                # Download Word Cloud
                buf = BytesIO()
                st.session_state['wordcloud'].save(buf, format="PNG", optimize=True, quality=100)
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Word Cloud as PNG",
                    data=byte_im,
                    file_name='wordcloud.png',
                    mime='image/png',
                )

                st.write("**Note:** After editing the frequency table, click the button below to regenerate the word cloud.")
            except AttributeError:
                st.write("Frequency table will appear here after generating.")
    
    # Button to generate word cloud from edited frequency table
    if not st.session_state['frequency_table'].empty:
        generate_wc_button = st.button("Generate Word Cloud from Edited Table")
        if generate_wc_button:
            try:
                st.session_state['wordcloud'] = generate_wordcloud(
                    st.session_state['frequency_table'],
                    mask_array,
                    customization_options
                )
                st.success("Word cloud generated from edited frequency table successfully!")
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
    
    # Display Word Cloud if available
    if st.session_state['wordcloud'] is not None:
        st.header("Generated Word Cloud")
        st.image(st.session_state['wordcloud'], use_container_width=True)
        
        # Download Word Cloud
        buf = BytesIO()
        st.session_state['wordcloud'].save(buf, format="PNG", optimize=True, quality=100)
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Word Cloud as PNG",
            data=byte_im,
            file_name='wordcloud.png',
            mime='image/png',
        )

# Run the main function
if __name__ == "__main__":
    main()

