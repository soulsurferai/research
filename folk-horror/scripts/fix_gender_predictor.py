# Test the correct API for global-gender-predictor
import global_gender_predictor as ggp

# Test some names
test_names = ['James', 'Mary', 'Kim', 'Leon', 'Louis', 'Kimberly']

for name in test_names:
    try:
        result = ggp.predict_gender(name)
        print(f"{name}: {result}")
    except Exception as e:
        print(f"Error with {name}: {e}")
