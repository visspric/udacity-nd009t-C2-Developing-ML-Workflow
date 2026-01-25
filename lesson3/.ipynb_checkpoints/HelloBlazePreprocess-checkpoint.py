import json
import os

# ----------------------------
# Functions
# ----------------------------

def label_data(input_data):
    """
    Label each review as helpful or unhelpful.
    """
    labeled_data = []
    HELPFUL_LABEL = "__label__1"
    UNHELPFUL_LABEL = "__label__2"
     
    with open(input_data, 'r') as f:
        for line in f:
            l_object = json.loads(line)
            helpful_votes = float(l_object['helpful'][0])
            total_votes = l_object['helpful'][1]
            review_text = l_object['reviewText']
            if total_votes != 0:
                if helpful_votes / total_votes > 0.5:
                    labeled_data.append(f"{HELPFUL_LABEL} {review_text}")
                elif helpful_votes / total_votes < 0.5:
                    labeled_data.append(f"{UNHELPFUL_LABEL} {review_text}")
    return labeled_data

def split_sentences(labeled_data):
    """
    Split each review into sentences, keeping the label.
    """
    new_split_sentences = []
    for d in labeled_data:
        label = d.split()[0]
        sentences = " ".join(d.split()[1:]).split(".")
        for s in sentences:
            s = s.strip()
            if s:  # skip empty sentences
                new_split_sentences.append(f"{label} {s}")
    return new_split_sentences

def write_data(data, train_path, test_path, proportion=0.9):
    """
    Split data into train and test files.
    """
    border_index = int(proportion * len(data))
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    with open(train_path, 'w') as train_f, open(test_path, 'w') as test_f:
        for i, d in enumerate(data):
            if i < border_index:
                train_f.write(d + '\n')
            else:
                test_f.write(d + '\n')

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # Detect JSON input file in SageMaker processing input directory
    input_dir = "/opt/ml/processing/input"
    input_file = None
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            input_file = os.path.join(input_dir, f)
            break

    if input_file is None:
        raise FileNotFoundError("No JSON input file found in /opt/ml/processing/input")

    # Preprocess
    labeled_data = label_data(input_file)
    split_data = split_sentences(labeled_data)

    # Write outputs
    write_data(
        split_data,
        "/opt/ml/processing/output/train/hello_blaze_train_scikit",
        "/opt/ml/processing/output/test/hello_blaze_test_scikit",
        proportion=0.9
    )

    print("Preprocessing completed successfully!")
