from data_loader import load_data
from data_validator import validate_data
from preprocessor import preprocess_data
from model_builder import build_logistic_regression
from trainer import train_model
from evaluator import evaluate_model

def main():
    df = load_data()
    df = validate_data(df)
    df = preprocess_data(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = build_logistic_regression()
    model, X_test, y_test = train_model(model, X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
