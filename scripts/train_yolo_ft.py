"""Fine-tune the paddle YOLO detector on in-domain labeled contact frames (warm start)."""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/paddle_yolo26s/weights/best.pt")
    p.add_argument("--data", default="datasets/paddle_ft/data.yaml")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="paddle_ft")
    args = p.parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
