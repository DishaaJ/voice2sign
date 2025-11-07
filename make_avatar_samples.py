import os, json, argparse, csv

def main(pose_dir, meta_csv, out_dir):
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Read metadata CSV file
    with open(meta_csv, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            base = os.path.splitext(row['filename'])[0]
            pfile = os.path.join(pose_dir, base + '_pose.json')

            if not os.path.exists(pfile):
                print("⚠️ Missing:", pfile)
                continue

            # Load pose file
            with open(pfile, 'r', encoding='utf8') as pf:
                pose = json.load(pf)

            # Ensure proper format (flatten frames)
            try:
                frame = pose['frames'][0]
                pose_vec = (
                    frame['pose_keypoints']
                    + frame['hand_left_keypoints']
                    + frame['hand_right_keypoints']
                    + frame['face_keypoints']
                )
            except Exception as e:
                print(f"⚠️ Error reading {pfile}: {e}")
                continue

            # Save cleaned training sample
            data = {'gloss': row['gloss'].split(), 'pose': pose_vec}
            out_path = os.path.join(out_dir, base + '.json')
            json.dump(data, open(out_path, 'w', encoding='utf8'))

            print("✅ Sample created:", base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate avatar training samples")
    parser.add_argument('--pose_dir', required=True, help="Path to folder containing pose JSON files")
    parser.add_argument('--meta', required=True, help="Path to metadata CSV file")
    parser.add_argument('--out', required=True, help="Output folder for samples")

    args = parser.parse_args()
    main(args.pose_dir, args.meta, args.out)
