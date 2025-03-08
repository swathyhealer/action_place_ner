import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
from collections import defaultdict


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def load_tsv(file_path):
        sentences = []
        sentence = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Non-empty line
                    res = line.strip().split("\t")
                    word = res[0]
                    tag = res[1:]
                    sentence.append((word, tag))
                else:  # Empty line indicates a new sentence
                    if sentence:
                        sentences.append(sentence)
                        sentence = []

        if sentence:  # Add the last sentence if file doesnt end with a newline
            sentences.append(sentence)

        return sentences

    @staticmethod
    def spacy_format_converter(data, label):
        B_label = "B-" + label
        I_label = "I-" + label
        target_labels = [B_label, I_label]
        converted_data = []
        for item in data:

            text = ""
            entities = []
            for word, word_labels in item:
                if text == "":
                    text = word
                    pointer = 0
                else:
                    text = text + " " + word
                    pointer = pointer + 1

                if B_label in word_labels:

                    start = pointer
                    end = len(word) + pointer  # actual +1
                    pointer = end
                    entities.append((start, end, label))
                elif I_label in word_labels:
                    if entities != [] and entities[-1][1] == pointer - 1:
                        end = len(word) + pointer
                        pointer = end
                        temp = entities.pop()

                        entities.append((temp[0], end, label))
                    else:
                        start = pointer
                        end = len(word) + pointer
                        pointer = end
                        entities.append((start, end, label))
                else:
                    end = len(word) + pointer
                    pointer = end

            converted_data.append((text, {"entities": entities}))
        return converted_data

    @staticmethod
    def train_val_test_splitter(data):
        train_no = int(len(data) * 0.6)
        test_no = int(len(data) * 0.2)
        training_data = data[:train_no]
        testing_data = data[train_no : train_no + test_no]
        val_data = data[train_no + test_no :]
        return training_data, testing_data, val_data

    @staticmethod
    def save_spacy_binary_format_data(spacy_format_data, out_path):
        nlp = spacy.blank("en")
        db = DocBin()

        for text, annotations in spacy_format_data:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations.get("entities"):
                span = doc.char_span(start, end, label=label)
                if span is None:
                    print(f"Skipping entity for '{text}': ({start}, {end}, '{label}')")
                else:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        # "./training_data.spacy"
        db.to_disk(out_path)

    @staticmethod
    def evaluate_ner_model(model_path, eval_data):

        nlp = spacy.load(model_path)
        scorer = Scorer()

        examples = []
        for text, annotations in eval_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        # Score all examples at once
        scores = scorer.score(examples)

        # Retrieve evaluation metrics
        precision = scores.get("token_p", 0.0) * 100
        recall = scores.get("token_r", 0.0) * 100
        f1_score = scores.get("token_f", 0.0) * 100
        accuracy = scores.get("token_acc", 0.0) * 100

        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1-Score: {f1_score:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

    @staticmethod
    def generate_legend(label_colors):

        legend_md = "| Entity Label       | Color       |\n"
        legend_md += "|--------------------|-------------|\n"
        for label, color in label_colors.items():
            legend_md += f"| {label} | <span style='background-color:{color};'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> |\n"
        return legend_md

    @staticmethod
    def group_entities_by_span(entities):
        span_dict = defaultdict(list)
        for entity in entities:
            span = (entity["start"], entity["end"], entity["text"])
            span_dict[span].append(entity["label"])
        return span_dict

    @staticmethod
    def generate_gradient_color(labels, label_colors):
        colors = [label_colors[label] for label in labels if label in label_colors]
        gradient = ", ".join(colors)
        return f"linear-gradient(90deg, {gradient})"

    @staticmethod
    def highlight_entities(text, grouped_entities, label_colors):
        highlighted_text = ""
        last_end = 0

        for (start, end, span_text), labels in sorted(grouped_entities.items()):

            highlighted_text += text[last_end:start]

            if len(labels) > 1:
                background_style = f"background: {Helper.generate_gradient_color(labels, label_colors)};"
            else:
                color = label_colors.get(labels[0], "#D3D3D3")
                background_style = f"background-color: {color};"

            highlighted_text += f"<span style='{background_style}'>{span_text}</span>"
            last_end = end

        highlighted_text += text[last_end:]

        return highlighted_text
