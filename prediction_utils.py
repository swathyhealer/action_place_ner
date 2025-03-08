import spacy


class PredictionPipeline:
    def __init__(self, model_path):
        self.model = spacy.load(model_path)

    def predict(self, text):
        doc = self.model(text)
        result = []
        for ent in doc.ents:
            ent_details = {
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "text": ent.text,
            }
            result.append(ent_details)
        return result


sample_text = "C/o throat pain since yesterday, generalized body pain, pain upper back . Also has itching , redness and swelling over both lower legs allegedly after mosquito bite ddx eczema o/e cvs s1s2, chest clear has tonsillar exudates left side,"


class NerPipeline:
    def __init__(self):
        self.activity_pipe = PredictionPipeline(
            model_path="./activity_output/model-best"
        )
        self.place_pipe = PredictionPipeline(model_path="./place_output/model-best")

    def process(self, text):
        activity_result = self.activity_pipe.predict(text)
        pipe_result = self.place_pipe.predict(text)
        result = activity_result + pipe_result
        return result


ner_pipeline = NerPipeline()
