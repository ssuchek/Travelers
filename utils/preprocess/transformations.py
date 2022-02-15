import config as constants
from .preprocessor import PreprocessTransformation

BASIC_PREPROCESS = [
    PreprocessTransformation("subcategory_prev", "subcategory_prev", "fill_na_with_value", ""),
    PreprocessTransformation("item_description", "item_description", "fill_na_with_value", ""),
    PreprocessTransformation("zip", "zip", "to_numeric")
]

CATEGORIES_WORD_PREPROCESS = [
    PreprocessTransformation("subcategory_prev", "subcategory_prev_processed", "identity"),
    PreprocessTransformation("subcategory_prev_processed", "subcategory_prev_processed", "remove_punctuation"),
    # PreprocessTransformation("subcategory_prev_processed", "subcategory_prev_processed", "remove_stop_words"),
    PreprocessTransformation("subcategory_prev_processed", "subcategory_prev_processed", "remove_verbs"),
    PreprocessTransformation("subcategory_prev_processed", "subcategory_prev_processed", "remove_one_letter_non_alphanumeric_words"),
    PreprocessTransformation("item_description", "item_description_processed", "identity"),
    PreprocessTransformation("item_description_processed", "item_description_processed", "remove_punctuation"),
    # PreprocessTransformation("item_description_processed", "item_description_processed", "remove_stop_words"),
    PreprocessTransformation("item_description_processed", "item_description_processed", "remove_verbs"),
    PreprocessTransformation("item_description_processed", "item_description_processed", "remove_one_letter_non_alphanumeric_words")
]

WEIGHTS_PREPROCESS = [
    PreprocessTransformation("waste_type", "waste_type", "fill_na_with_value", ""),
    PreprocessTransformation("primary_desc", "primary_desc", "fill_na_with_value", ""),
    PreprocessTransformation("secondary_desc", "secondary_desc", "fill_na_with_value", ""),
    PreprocessTransformation("material", "material", "fill_na_with_value", ""),
    PreprocessTransformation("dimensions", "dimensions", "fill_na_with_value", ""),
    PreprocessTransformation("values_desc", "values_desc", "fill_na_with_value", ""),
    PreprocessTransformation("unit", "unit", "fill_na_with_value", ""),
    PreprocessTransformation("weight_lbs", "weight_lbs", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("weight_ustons", "weight_ustons", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("volume_cf", "volume_cf", "fill_na_and_negatives", 0, 0),
    PreprocessTransformation("volume_cy", "volume_cy", "fill_na_and_negatives", 0, 0)
]

WEIGHTS_WORD_PREPROCESS = [
    PreprocessTransformation("primary_desc", "primary_desc_processed", "identity"),
    PreprocessTransformation("primary_desc_processed", "primary_desc_processed", "remove_punctuation"),
    # PreprocessTransformation("primary_desc_processed", "primary_desc_processed", "remove_stop_words"),
    PreprocessTransformation("primary_desc_processed", "primary_desc_processed", "remove_verbs"),
    PreprocessTransformation("primary_desc_processed", "primary_desc_processed", "remove_one_letter_non_alphanumeric_words", ["&", ";"]),
    PreprocessTransformation("secondary_desc", "secondary_desc_processed", "identity"),
    PreprocessTransformation("secondary_desc_processed", "secondary_desc_processed", "remove_punctuation"),
    # PreprocessTransformation("secondary_desc_processed", "secondary_desc_processed", "remove_stop_words"),
    PreprocessTransformation("secondary_desc_processed", "secondary_desc_processed", "remove_verbs"),
    PreprocessTransformation("secondary_desc_processed", "secondary_desc_processed", "remove_one_letter_non_alphanumeric_words", ["&", ";"]),
    PreprocessTransformation("material", "material_processed", "identity"),
    PreprocessTransformation("material_processed", "mamaterial_processedterial", "remove_punctuation"),
    # PreprocessTransformation("material_processed", "material_processed", "remove_stop_words"),
    PreprocessTransformation("material_processed", "material_processed", "remove_verbs"),
    PreprocessTransformation("material_processed", "material_processed", "remove_one_letter_non_alphanumeric_words", ["&", ";"]),
    PreprocessTransformation("dimensions", "dimensions_processed", "identity"),
    PreprocessTransformation("dimensions_processed", "dimensions_processed", "remove_punctuation"),
    # PreprocessTransformation("dimensions_processed", "dimensions_processed", "remove_stop_words"),
    PreprocessTransformation("dimensions_processed", "dimensions_processed", "remove_verbs"),
    PreprocessTransformation("dimensions_processed", "dimensions_processed", "remove_one_letter_non_alphanumeric_words", ["&", ";"]),
    PreprocessTransformation("values_desc", "values_desc_processed", "identity"),
    PreprocessTransformation("values_desc_processed", "values_desc_processed", "remove_punctuation"),
    # PreprocessTransformation("values_desc_processed", "values_desc_processed", "remove_stop_words"),
    PreprocessTransformation("values_desc_processed", "values_desc_processed", "remove_verbs"),
    PreprocessTransformation("values_desc_processed", "values_desc_processed", "remove_one_letter_non_alphanumeric_words", ["&", ";"])
]

WEIGHTS_WORD_FREQUENCY_PREPROCESS = [
    PreprocessTransformation("item_description", "item_description", "remove_punctuation"),
    PreprocessTransformation("item_description", "item_description", "remove_stop_words", constants.FREQUENCY_STOP_WORDS),
    PreprocessTransformation("item_description", "item_description", "remove_verbs"),
    PreprocessTransformation("item_description", "item_description", "remove_one_letter_words"),
    PreprocessTransformation("item_description", "item_description", "remove_numeric_words")
]