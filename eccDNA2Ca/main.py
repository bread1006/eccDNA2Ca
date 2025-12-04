import argparse
import pandas as pd
from utils import eccDNA2Ca

def main():
    parser = argparse.ArgumentParser(description="Predict eccDNA cancer association using the eccDNA2Ca model.")
    parser.add_argument("--fasta_file", type=str, help="Path to the FASTA file containing eccDNA sequences.")
    parser.add_argument("--xgb_features_file", type=str, help="Path to the Excel file containing features for XGBoost.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing the trained models.")

    args = parser.parse_args()

    # Create model instance
    model = eccDNA2Ca(model_dir=args.model_dir)

    # Load XGBoost features if provided
    xgb_features_df = None
    if args.xgb_features_file:
        if args.xgb_features_file.endswith(".csv"):
        xgb_features_df = pd.read_csv(args.xgb_features_file)
     else:
        xgb_features_df = pd.read_excel(args.xgb_features_file)

    # Run prediction
    predictions = model.predict(fasta_file=args.fasta_file, xgb_features_df=xgb_features_df)

    # Output predictions
    print("ID\Predictions:")
    for i, prob in enumerate(predictions, 1):
        print(f"seq{i}\t {prob:.4f}")

if __name__ == "__main__":
    main()
