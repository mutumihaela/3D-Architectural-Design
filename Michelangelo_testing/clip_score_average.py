# clip_score_average.py

def compute_and_append_average(file_path):
    scores = []

    # Read and extract scores
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                try:
                    score = float(line.split(':')[1].strip())
                    scores.append(score)
                except ValueError:
                    print(f"Could not parse score from line: {line.strip()}")

    # Compute and append average
    if scores:
        average = sum(scores) / len(scores)
        result_line = f"\nAverage_CLIP_Score: {average:.6f}\n"
        with open(file_path, 'a') as f:
            f.write(result_line)
        print(f"Average CLIP Score: {average:.6f} (also saved to file)")
    else:
        print("No valid scores found in the file.")

# Run the function
compute_and_append_average("clip_scores.txt")
