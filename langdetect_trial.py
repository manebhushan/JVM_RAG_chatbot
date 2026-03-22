from langdetect import detect 
import re
from lingua import Language, LanguageDetectorBuilder

# def is_english(text: str) -> bool:
#     """
#     Checks if a given string is in English.

#     Args:
#         text: The input string to check.

#     Returns:
#         True if the detected language is English, False otherwise.
#     """
#     # Langdetect can struggle with very short or non-alphabetic text.
#     # We preprocess the text to ensure it's suitable for detection.
#     # Remove special characters and numbers, and convert to lowercase.
#     clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    
#     # If the cleaned text is too short, langdetect may not be reliable.
#     # We can handle this by returning False or using a different logic.
#     if len(clean_text.split()) < 3: # A simple heuristic for short text
#         return False

#     try:
#         # Detect the language of the cleaned text
#         lang_code = detect(clean_text)
        
#         # Check if the detected language code is 'en' for English
#         return lang_code == 'en'
#     except Exception:
#         # If an error occurs (e.g., text is empty or non-alphabetic),
#         # we can't reliably detect the language.
#         return False



# print(is_english("does praying to god before exam will get me good marks?"))



detector = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models() # This is the default setting
    .with_minimum_relative_distance(0.9) # A higher value increases confidence
    .build()
)

def is_english(text: str) -> bool:
    """
    Checks if a given string is in English using the lingua-py library.
    
    Args:
        text: The input string to check.

    Returns:
        True if the detected language is English, False otherwise.
    """
    # Preprocessing to handle special characters and numbers, which can confuse some detectors
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text["question"]).lower().strip()
    print(clean_text)
    # If the text is empty or too short, it's difficult to make a reliable prediction
    if not clean_text or len(clean_text.split()) < 2:
        return False
    
    # Use the pre-built detector to detect the language
    detected_language = detector.detect_language_of(clean_text)
    print(detected_language)
    # Return True if the detected language is English
    return detected_language == Language.ENGLISH


# print(is_english({"question":"जीवंनविद्या विद्यार्थी दशेत कशी लागू करावी?"}))
# print(is_english({"question":"does praying to god before exam will get me good marks?"}))
print(is_english({"question":"how to stay away from addictions"}))