import pandas as pd
import click
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


@click.command()
@click.option('--ground_truth_csv', default="data/NTU_RGB+D/NTU_RGB+D_annotations.csv", help="Path to the ground truth CSV.")
@click.option('--predictions_csv', default="data/NTU_RGB+D/NTU_RGB+D_predictions.csv", help="Path to the predictions CSV.")
@click.option('--output_dir', default="results/", help="Directory to save evaluation results.")
def evaluate_predictions(ground_truth_csv, predictions_csv, output_dir):
    """
    모델 예측값과 Ground Truth 비교, 성능 평가
    """
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predictions_csv)

    gt_df["label"] = gt_df["label"].str.lower().str.replace(" ", "_")
    pred_df["predicted_label"] = pred_df["predicted_label"].str.lower().str.replace(" ", "_")

    common_videos = set(gt_df["video_id"]) & set(pred_df["video_id"])
    gt_df = gt_df[gt_df["video_id"].isin(common_videos)]
    pred_df = pred_df[pred_df["video_id"].isin(common_videos)]

    merged_df = pd.merge(gt_df, pred_df, on="video_id", how="inner")
    merged_df.rename(columns={"label": "ground_truth", "predicted_label": "prediction"}, inplace=True)

    y_true = merged_df["ground_truth"]
    y_pred = merged_df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, digits=4, zero_division=1)
    
    unique_labels = sorted(set(y_true) | set(y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)

    results_file = f"{output_dir}/classification_report.txt"
    with open(results_file, "w") as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

    print(f"Classification report saved to {results_file}")

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=y_true.unique(), yticklabels=y_true.unique())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.show()

    print(f"Confusion Matrix saved to {output_dir}/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_predictions()
