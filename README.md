# Named Entity Recognition (NER) Project - NERA

This project focuses on implementing a Named Entity Recognition (NER) system capable of identifying multiple labels within text, specifically "ACTIVITY OR CAUSE" and "PLACE". Given the multilabel nature of the dataset, we employ two separate spaCy models to handle each label independently and then combine their outputs for comprehensive entity recognition.

**NOTE :** After cloning repo , run flowing command to ensure LFS files are downloaded properly.

   ```
      git lfs pull
   ```

## Approach

1. **Data Preparation**:
   - **Annotation**: The dataset is annotated with entities corresponding to "ACTIVITY OR CAUSE" and "PLACE".
   - **Preprocessing**: Using `preprocess.py`, the annotated data is converted into a format suitable for spaCy training, resulting in training, validation, and testing datasets.

2. **Model Training**:
   - **Separate Models**: Two distinct spaCy models are trained:
     - **Activity NER Model**: Trained using `activity_config.cfg`.
     - **Place NER Model**: Trained using `place_config.cfg`.
   - **Training Command**: Each model is trained using the command:
     ```
     python -m spacy train <config_file> --output <output_directory>
     ```
     For example:
     ```
     python -m spacy train activity_config.cfg --output ./activity_output
     ```

3. **Model Evaluation**:
   - **Evaluation Script**: The `evaluate_model.py` script assesses the performance of both models on the test dataset.
   - **Metrics**: Standard NER evaluation metrics such as precision, recall, and F1-score are computed.

4. **Prediction Pipeline**:
   - **NerPipeline Class**: Located in `prediction_utils.py`, this class loads both trained models and provides methods for predicting entities in new text data.
   - **Combining Outputs**: Predictions from both models are merged to handle overlapping or conflicting entity spans.

5. **Demo Application**:
   - **Streamlit App**: A user-friendly interface is built using Streamlit (`app.py`) to demonstrate the NER system's capabilities. Users can input text and visualize the recognized entities.

## Usage

1. **Data Preprocessing**:
   ```
   python preprocess.py
   ```


2. **Training Models**:
   - **Activity NER Model**:
     ```
     python -m spacy train activity_config.cfg --output ./activity_output
     ```
   - **Place NER Model**:
     ```
     python -m spacy train place_config.cfg --output ./place_output
     ```

3. **Evaluating Models**:
   ```
   python evaluate_model.py
   ```


4. **Running the Streamlit App**:
   ```
   streamlit run app.py
   ```


## Notes

- **Model Locations**:
  - Activity Model: `./activity_output/best_model`
  - Place Model: `./place_output/best_model`



This structured approach ensures that each entity type is accurately recognized by its respective model, and their combined outputs provide a comprehensive NER system. 