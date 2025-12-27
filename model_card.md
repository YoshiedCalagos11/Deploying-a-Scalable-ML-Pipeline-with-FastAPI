# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a supervised machine learning classification model trained to predict whether an individual earns more than $50K per year based on U.S. Census demographic and employment data. The model uses a Random Forest classifier implemented with scikit-learn. Categorical features are encoded using one-hot encoding, and numerical features are used as-is.

## Intended Use

The intended use of this model is for educational and demonstration purposes only. It is designed to showcase how to build, train, evaluate, and deploy a machine learning model using a scalable pipeline and a RESTful API. The model should not be used for real-world financial, hiring, or income-related decision making.

## Training Data

The model was trained using the U.S. Census Income dataset. The dataset contains demographic and employment-related features such as age, education, occupation, marital status, hours worked per week, and native country. The target variable indicates whether an individual's income exceeds $50K per year.

## Evaluation Data

The evaluation data consists of a held-out test split from the same U.S. Census dataset used for training. The dataset was split into training and testing sets to evaluate the model’s performance on unseen data.

## Metrics

The model is evaluated using classification metrics including precision, recall, and F1-score. Overall model performance metrics are printed during training. In addition, model performance is evaluated on slices of the data across categorical features, and these results are stored in `slice_output.txt`.

## Ethical Considerations

This model uses demographic data, which may encode historical biases related to income, gender, race, and occupation. Predictions from this model may reflect or amplify these biases. Care should be taken to avoid using the model in any context where it could negatively impact individuals or groups.

## Caveats and Recommendations

The model is limited by the quality and scope of the training data and does not account for external economic factors or changes over time. Model predictions should not be interpreted as definitive or causal. Future improvements could include bias analysis, fairness metrics, and evaluation on more recent or diverse datasets.
