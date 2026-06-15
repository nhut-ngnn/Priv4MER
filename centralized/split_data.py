import pickle, random, argparse, os

def main(args):
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    random.shuffle(data)

    split_idx = int((1 - args.ratio) * len(data))
    label_data = data[:split_idx]
    unlabel_data = data[split_idx:]

    for item in label_data:
        item["is_pseudo"] = False
        item["confidence"] = 1.0

    for item in unlabel_data:
        item["is_pseudo"] = True
        item["confidence"] = 0.0

    if args.label_output is None or args.unlabel_output is None:
        base, ext = os.path.splitext(args.input)
        args.label_output = args.label_output or f"{base}_label_{1-args.ratio:.2f}{ext}"
        args.unlabel_output = args.unlabel_output or f"{base}_unlabel_{args.ratio:.2f}{ext}"

    with open(args.label_output, "wb") as f:
        pickle.dump(label_data, f)

    with open(args.unlabel_output, "wb") as f:
        pickle.dump(unlabel_data, f)

    print(f"Split done: Label={len(label_data)} | Unlabel={len(unlabel_data)}")
    print(f"   Saved: {args.label_output}, {args.unlabel_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into label and unlabel parts")
    parser.add_argument("--input", type=str, required=True, help="Path to input pickle file")
    parser.add_argument("--label_output", type=str, default=None, help="Output file for labeled data")
    parser.add_argument("--unlabel_output", type=str, default=None, help="Output file for unlabeled data")
    parser.add_argument("--ratio", type=float, default=0.3, help="Ratio of unlabeled data (0.0 - 1.0)")
    args = parser.parse_args()
    main(args)
