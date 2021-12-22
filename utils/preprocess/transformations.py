from .preprocessor import PreprocessTransformation

BASIC_PREPROCESS = [
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "fill_na_with_value", ""),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "strip_values"),
    PreprocessTransformation("item_description", "item_description", "fill_na_with_value", ""),
    PreprocessTransformation("item_description", "item_description", "strip_values"),
    PreprocessTransformation("zip", "zip", "to_numeric")
]

BASIC_WORD_PREPROCESS = [
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_punctuation"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_stop_words"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_verbs"),
    # PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_non_alphabetical_words"),
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("item_description", "item_description", "remove_punctuation"),
    PreprocessTransformation("item_description", "item_description", "remove_stop_words"),
    PreprocessTransformation("item_description", "item_description", "remove_verbs"),
    # PreprocessTransformation("item_description", "item_description", "remove_non_alphabetical_words"),
    PreprocessTransformation("item_description", "item_description", "remove_one_letter_non_alphanumeric_words"),
]

WEIGHTS_PREPROCESS = [
    PreprocessTransformation("waste_type", "waste_type", "fill_na_with_value", ""),
    PreprocessTransformation("primary_desc", "primary_desc", "fill_na_with_value", ""),
    PreprocessTransformation("secondary_desc", "secondary_desc", "fill_na_with_value", ""),
    PreprocessTransformation("material", "material", "fill_na_with_value", ""),
    PreprocessTransformation("dimensions", "dimensions", "fill_na_with_value", ""),
    #PreprocessTransformation("tertiary_desc", "tertiary_desc", "fill_na_with_value", ""),
    PreprocessTransformation("values_desc", "values_desc", "fill_na_with_value", ""),
    PreprocessTransformation("unit", "unit", "fill_na_with_value", ""),
    PreprocessTransformation("weight_lbs", "weight_lbs", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("weight_ustons", "weight_ustons", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("volume_cf", "volume_cf", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("volume_cy", "volume_cy", "fill_na_and_negatives", 0, 0)
]

# CATEGORIES_WORD_PREPROCESS = [
#     PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_punctuation"),
#     PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_stop_words"),
#     PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_verbs"),
#     PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_non_alphabetical_words"),
#     PreprocessTransformation("subcategory_prev", "subcategory_prev", "remove_one_letter_non_numeric_words"),
#     PreprocessTransformation("item_description", "item_description", "remove_punctuation"),
#     PreprocessTransformation("item_description", "item_description", "remove_stop_words"),
#     PreprocessTransformation("item_description", "item_description", "remove_verbs"),
#     PreprocessTransformation("item_description", "item_description", "remove_non_alphabetical_words"),
#     PreprocessTransformation("item_description", "item_description", "remove_one_letter_non_numeric_words"),
# ]

WEIGHTS_WORD_PREPROCESS = [
    PreprocessTransformation("primary_desc", "primary_desc", "remove_punctuation"),
    PreprocessTransformation("primary_desc", "primary_desc", "remove_stop_words"),
    PreprocessTransformation("primary_desc", "primary_desc", "remove_verbs"),
    PreprocessTransformation("primary_desc", "primary_desc", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("secondary_desc", "secondary_desc", "remove_punctuation"),
    PreprocessTransformation("secondary_desc", "secondary_desc", "remove_stop_words"),
    PreprocessTransformation("secondary_desc", "secondary_desc", "remove_verbs"),
    PreprocessTransformation("secondary_desc", "secondary_desc", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("material", "material", "remove_punctuation"),
    PreprocessTransformation("material", "material", "remove_stop_words"),
    PreprocessTransformation("material", "material", "remove_verbs"),
    PreprocessTransformation("material", "material", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("dimensions", "dimensions", "remove_punctuation"),
    PreprocessTransformation("dimensions", "dimensions", "remove_stop_words"),
    PreprocessTransformation("dimensions", "dimensions", "remove_verbs"),
    PreprocessTransformation("dimensions", "dimensions", "remove_one_letter_non_alphanumeric_words"),
    # PreprocessTransformation("tertiary_desc", "tertiary_desc", "remove_punctuation"),
    # PreprocessTransformation("tertiary_desc", "tertiary_desc", "remove_stop_words"),
    # PreprocessTransformation("tertiary_desc", "tertiary_desc", "remove_verbs"),
    # PreprocessTransformation("tertiary_desc", "tertiary_desc", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("values_desc", "values_desc", "remove_punctuation"),
    PreprocessTransformation("values_desc", "values_desc", "remove_stop_words"),
    PreprocessTransformation("values_desc", "values_desc", "remove_verbs"),
    PreprocessTransformation("values_desc", "values_desc", "remove_one_letter_non_alphanumeric_words")
]