# Babak-Yousefian-BD-T2
---


# 📘 MinHash-Based Document Similarity Estimator

## 📌 Assignment: تمرین شماره ۲ – داده‌های حجیم

### 📂 Description
This Python project analyzes the similarity between multiple text documents using both:

- **Exact Jaccard Similarity**
- **Approximate MinHashing**

It also visualizes the resulting similarity matrices in a 2D heatmap using **matplotlib**.

---

## 📁 Files

- `Doc1.txt`: Contains sample documents (`doc1` to `doc4`) in Python dictionary format.
- `main.py`: Main program that processes the documents.
- (Optional) `requirements.txt`: Python packages used in the project.

---

## 🚀 How to Run

```bash
pip install matplotlib
python main.py
```

Ensure that `Doc1.txt` is in the same directory as `main.py`.

---

## 📖 Detailed Explanation (Line by Line)

### 1. **Imports**
```python
import random
import matplotlib.pyplot as plt
import itertools
```
- `random`: Generates random coefficients for hash functions.
- `matplotlib.pyplot`: Used to visualize similarity matrices as heatmaps.
- `itertools`: (Unused in final version but imported for potential future enhancements).

### 2. **Loading the Document File**
```python
def loadData(filename="Doc1.txt"):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()
    return eval(data)
```
- Reads the `Doc1.txt` file using `with open(...)`.
- Converts it to a dictionary using `eval()`.

### 3. **Shingling Documents**
```python
def get_shingles(text, k=3):
    text = text.replace(" ", "")
    return set(text[i:i+k] for i in range(len(text) - k + 1))
```
- Converts text into a set of 3-character substrings (shingles).

### 4. **Exact Jaccard Similarity**
```python
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)
```
- Measures the similarity based on intersection over union.

### 5. **Hash Function Generator**
```python
def generate_hash_functions(num_funcs, max_shingle_id, prime):
    ...
```
- Creates MinHash functions of the form `(a*x + b) % prime`.

### 6. **Building MinHash Signatures**
```python
def build_signature_matrix(shingle_sets, hash_funcs, shingle_id_map):
    ...
```
- Builds the signature matrix for all documents.

### 7. **Estimated Similarity from Signatures**
```python
def estimated_similarity(sig1, sig2):
    ...
```
- Compares two signature vectors.

### 8. **Plotting the Heatmap**
```python
def plot_similarity_matrix(matrix, labels, title):
    ...
```
- Visualizes similarity matrix using `matplotlib`.

### 9. **Main Program**
```python
def main():
    ...
```
- Executes the full pipeline of loading, comparing, and plotting.

### 10. **Entry Point**
```python
if __name__ == "__main__":
    main()
```

---

## 🧪 Sample Output (CLI)
```text
=== Exact Jaccard Similarity Matrix ===
['1.0000', '0.4138', '0.0000', '0.3448']
['0.4138', '1.0000', '0.0000', '0.2763']
['0.0000', '0.0000', '1.0000', '0.0000']
['0.3448', '0.2763', '0.0000', '1.0000']

=== Estimated Similarity Matrix ===
['1.0000', '0.4500', '0.0500', '0.3500']
['0.4500', '1.0000', '0.0000', '0.3000']
['0.0500', '0.0000', '1.0000', '0.0000']
['0.3500', '0.3000', '0.0000', '1.0000']
```

---

## 🎯 Features

| Feature                         | Implemented |
|----------------------------------|-------------|
| Load documents from file         | ✅           |
| Exact Jaccard similarity         | ✅           |
| Randomized MinHash functions     | ✅           |
| Signature matrix generation      | ✅           |
| Approximate similarity (MinHash) | ✅           |
| Matrix visualization (2D)        | ✅           |

---

## 🔒 Security Note

Replace `eval()` with `ast.literal_eval()` for security:
```python
from ast import literal_eval
data = literal_eval(file.read())
```

---

## 📦 Dependencies

```text
matplotlib
numpy
pandas
autopep8
pip
setuptools
build
```

Install with:
```bash
python.exe -m pip install --upgrade pip autopep8 build setuptools pandas numpy matplotlib
```
---
# 📌 Imports

```bash
import random
import matplotlib.pyplot as plt
import re
from ast import literal_eval
```

    random: Used to generate random a and b values for our MinHash functions.

    matplotlib.pyplot as plt: A library for plotting graphs. Here, it creates a heatmap to visualize similarity.

    re: Regular expressions. Helps us extract only the dictionary part from the Doc1.txt content.

    literal_eval: Safer version of eval(). Converts string to dictionary without executing arbitrary code.
---

## 📌 Load Data from File

```bash
def loadData(filename="Doc1.txt"):
```

Defines a function named loadData() that takes a file name (default is "Doc1.txt"). It returns a dictionary of documents.

```bash
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()
```

    Opens the file using a context manager (with open) to ensure it closes automatically.

    Reads the content of the file as a string.

```bash
    match = re.search(r"documents\\s*=\\s*(\\{.*\\})", data, re.DOTALL)
```

    Uses regex to find text that matches: documents = { ... }.

    \\s* means "zero or more spaces".

    (\\{.*\\}) matches the dictionary itself.

    re.DOTALL: makes . match newlines too (needed for multi-line dictionaries).

```bash
    if match:
        return literal_eval(match.group(1))
```

    If the match was successful, convert the string dictionary to a real Python dictionary.

    match.group(1) gets just the {...} part.

    literal_eval() ensures safety.

```bash
    else:
        raise ValueError("Could not extract dictionary from file")
```

    Raises an error if the expected dictionary wasn't found in the file.

---

## 📌 Shingling Function

```bash
def get_shingles(text, k=3):
```

Defines a function to convert text into shingles (small substrings). k=3 means 3 characters per shingle.

```bash
    text = text.replace(" ", "")
```
    Removes all spaces to create continuous substrings.

```bash
    return set(text[i:i+k] for i in range(len(text) - k + 1))
```
    Uses a loop to get all substrings of length 3.

    Returns them as a set (to remove duplicates).
---

## 📌 Exact Jaccard Similarity

```bash
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
```

    Calculates Jaccard Similarity = intersection size ÷ union size.

    set1 & set2: elements in both.

    set1 | set2: elements in either.

    If both sets are empty, returns 0.0 to avoid division by zero.
---

## 📌 Generate Hash Functions for MinHashing

```bash
def generate_hash_functions(num_funcs, max_shingle_id, prime):
```

Defines a function that generates num_funcs random hash functions of the form (a * x + b) % prime.

```bash
    hash_funcs = []
```
    Initializes an empty list to store the functions.

```bash
    for _ in range(num_funcs):
        a = random.randint(1, prime - 1)
        b = random.randint(0, prime - 1)
```

    Random values a and b for the hash function.

    a must be ≥ 1, b can be 0.

```bash
        hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % prime)
```

    Adds a lambda function to the list.

    lambda x is a short way to define a function.

    a=a and b=b fix the values at the time of creation (important!).

```bash
    return hash_funcs
```

    Returns the list of hash functions.

---

## 📌 MinHash Signature Matrix

```bash
def build_signature_matrix(shingle_sets, hash_funcs, shingle_id_map):
```

Builds a matrix that stores the minimum hash value for each document using each hash function.

```bash
    num_docs = len(shingle_sets)
    num_hashes = len(hash_funcs)
    signatures = [[float('inf')] * num_docs for _ in range(num_hashes)]
```

    Initializes a 2D matrix with ∞ (largest value possible) so it can be minimized later.

```bash
    for shingle, shingle_id in shingle_id_map.items():
```

    Goes through all shingles and their unique IDs.

```bash
        for doc_idx, doc_shingles in enumerate(shingle_sets):
            if shingle in doc_shingles:
```

    For each document, check if this shingle exists.

```bash
                for func_idx, h in enumerate(hash_funcs):
                    hash_val = h(shingle_id)
```
    Apply each hash function to the current shingle ID.

```bash
                    if hash_val < signatures[func_idx][doc_idx]:
                        signatures[func_idx][doc_idx] = hash_val
```

    Update the signature if this hash value is smaller (MinHash logic).

```bash
    return signatures
```

    Return the full signature matrix.
---

## 📌 MinHash Estimated Similarity


```bash
def estimated_similarity(sig1, sig2):
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1) if sig1 else 0.0
```

    Compares two MinHash signatures.

    Similarity = % of hash values that are equal.

    Uses zip() to pair elements and counts matches.
---

## 📌 Plot Similarity Matrix

```bash
def plot_similarity_matrix(matrix, labels, title):
```

    Creates a heatmap (colored grid) from a 2D matrix.

```bash
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
```

    Initializes the figure and plots the matrix with color mapping (viridis).

```bash
    plt.colorbar(label='Similarity')
```

    Adds a color legend.

```bash
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
```

    Labels the X and Y axes with document names.

```bash
    plt.title(title)
```

    Sets the plot title.

```bash
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', color='white')
```

    Adds similarity numbers inside each cell.

```bash
    plt.tight_layout()
    plt.show()
```

    Adjusts layout and shows the plot.
---

## 📌 Main Function

```bash
def main():
```

    Main driver function.

```bash
    documents = loadData("Doc1.txt")
```

    Loads the document dictionary from file.

```bash
    k = 3
    doc_names = list(documents.keys())
    shingle_sets = [get_shingles(documents[doc], k) for doc in doc_names]
```

    Sets shingle length to 3.

    Gets document names and their corresponding shingle sets.

```bash
    all_shingles = set.union(*shingle_sets)
    shingle_id_map = {shingle: idx for idx, shingle in enumerate(all_shingles)}
```

    Collects all unique shingles and assigns them numeric IDs.

```bash
    max_shingle_id = len(shingle_id_map)
    prime = 4294967311
    num_hashes = 20
    hash_funcs = generate_hash_functions(num_hashes, max_shingle_id, prime)
```

    Sets parameters and generates 20 MinHash functions.

---

## 📌 Compute Exact Jaccard Similarity

```bash
    exact_matrix = [[0.0]*len(doc_names) for _ in range(len(doc_names))]
```

    Prepares empty matrix.

```bash
    for i in range(len(doc_names)):
        for j in range(len(doc_names)):
            exact_matrix[i][j] = jaccard_similarity(shingle_sets[i], shingle_sets[j])
```

    Computes and stores exact Jaccard similarity between all pairs.

```bash
    print("\\n=== Exact Jaccard Similarity Matrix ===")
    for row in exact_matrix:
        print(["{:.4f}".format(x) for x in row])
    plot_similarity_matrix(exact_matrix, doc_names, "Exact Jaccard Similarity")
```

    Prints matrix and plots it.

---

## 📌 Compute MinHash Similarity

```bash
    signatures = build_signature_matrix(shingle_sets, hash_funcs, shingle_id_map)
```

    Builds signature matrix.

```bash
    est_matrix = [[0.0]*len(doc_names) for _ in range(len(doc_names))]
    for i in range(len(doc_names)):
        for j in range(len(doc_names)):
            sig1 = [signatures[h][i] for h in range(num_hashes)]
            sig2 = [signatures[h][j] for h in range(num_hashes)]
            est_matrix[i][j] = estimated_similarity(sig1, sig2)
```

    Compares signature vectors and builds estimated similarity matrix.

```bash
    print("\\n=== Estimated Similarity Matrix ===")
    for row in est_matrix:
        print(["{:.4f}".format(x) for x in row])
    plot_similarity_matrix(est_matrix, doc_names, "Estimated Similarity (MinHash)")
```

    Prints and plots the estimated similarity matrix.
---

## 📌 Entry Point

```bash
if __name__ == "__main__":
    main()
```

    Runs the main() function only if this file is executed directly.
---
# @Author by : babak yousefian
---
