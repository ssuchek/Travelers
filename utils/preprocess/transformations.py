from .preprocessor import PreprocessTransformation

BASIC_PREPROCESS = [
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "fill_na_with_value", ""),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_punctuation"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_stop_words"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_non_alphabetical_words"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_one_letter_words"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "strip_values"),
    PreprocessTransformation("item_description", "item_description", "fill_na_with_value", ""),
    PreprocessTransformation("item_description", "item_description", "remove_punctuation"),
    PreprocessTransformation("item_description", "item_description", "remove_stop_words"),
    PreprocessTransformation("item_description", "item_description", "remove_non_alphabetical_words"),
    PreprocessTransformation("item_description", "item_description", "remove_one_letter_words"),
    PreprocessTransformation("item_description", "item_description", "strip_values"),
    PreprocessTransformation("zip", "zip", "to_numeric")
]