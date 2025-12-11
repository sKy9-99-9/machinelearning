# Convolution & CNNs — Assignment Notebook

This notebook is a hands-on tour from basic image filtering to a working Convolutional Neural Network (CNN).  
You’ll implement classic image operations (sepia, grayscale, blur), write your own **convolution** and **pooling**, apply real filters (Gaussian, sharpen, edges), and finish with both a **hand-coded feature extractor** and a **Keras CNN** on **Fashion-MNIST**.

---

## 🧰 Setup & Libraries

You’ll use:
- `numpy`, `matplotlib`, `math`
- `cv2` (OpenCV) for image I/O
- `scipy.signal.convolve` (fast convolution helper)
- `skimage.measure.block_reduce` (fast pooling helper)
- `tensorflow.keras` (for dense NN and CNN)
- Dataset: **Fashion-MNIST** via `tensorflow.keras.datasets.fashion_mnist` (10 clothing classes, 28×28 grayscale)

Images are treated as NumPy arrays:
- Color image shape: `(H, W, 3)` with RGB channels.
- Values are floats in `[0, 1]` unless stated otherwise.

---

## 🧭 Learning Path

1) Build intuition with pixel manipulations (smiley, sepia, grayscale).  
2) Implement blur with **patch averaging**.  
3) Generalize to **convolution** and **apply kernels**.  
4) Use real filters (Gaussian, sharpen, edges).  
5) Implement **max pooling**.  
6) Build a dense **baseline NN**.  
7) Create feature maps and pooling to form a **hand-coded CNN**.  
8) Implement a learnable **Keras CNN** and compare.

---

## ❤️ Warm-up & Filters

### Assignment 0 — Recreate the Red Smiley
**Goal:** Work with RGB arrays directly.

**What to do:**
- You’re given an 8×8×3 array `smiley` (initialized to ones).
- Set the listed coordinates in `pixel_list` to **red** by manipulating channels so that pixel `(r, c)` has `[1, 0, 0]`.
- Display with `plt.imshow(smiley)`.

---

### Assignment 1 — Sepia
**Function:** `def sepia(image): -> sepia_image`

**Goal:** Apply the classic sepia transform per pixel.

**What to implement:**
- For each pixel `(R,G,B)` compute:



- **Cap** each channel to `1.0` if it exceeds 1.
- Return an array with same shape as input.
- Input image is RGB; keep outputs in `[0, 1]`.

---

### Assignment 2 — Grayscale
**Function:** `def grayscale(image): -> gray_image`

**Goal:** Convert RGB to single-channel grayscale.

**What to implement:**
- Compute the **average** across the 3 channels for each pixel.
- Output shape `(H, W)` and values in `[0, 1]`.
- Assertions check min/max ∈ `[0,1]`.

---

### Assignment 3 — Blur (Box Blur on Grayscale)
**Function:** `def blur(image, size=3): -> blur_image`

**Goal:** Average each **size×size** patch to blur the image.

**What to implement:**
- For each output pixel, take the mean of the centered `size×size` patch.
- **Border handling:** choose **padding** or **cropping** (either is acceptable, document your choice).
- Input is **grayscale** (2D). `size` is an odd integer (e.g., 3, 5, 7).
- Return the blurred image; larger `size` ⇒ stronger blur.

**Hints:** Loop over **output** pixels; use NumPy slicing and `np.mean`.

---

## 🔢 Convolution

### Assignment 4a — Convolution (Matrix–Matrix)
**Function:** `def convolution(x, h): -> scalar`

**Goal:** Implement the mathematical convolution on a **single patch**.

**What to implement:**
- `x` and `h` are same-size 2D arrays.
- Return `sum_{i,j} x[i,j] * h[i,j]` (elementwise multiply & sum).
- This is the building block used by `convolve`.

---

### Assignment 4b — Convolve (Image–Kernel)
**Function:** `def convolve(image, h): -> out_image`

**Goal:** Slide a kernel over an image using your `convolution`.

**What to implement:**
- Input `image` is **grayscale** 2D; `h` is a **square** kernel.
- For each valid location, extract the centered patch and call `convolution(patch, h)`.
- Place the result at the **center** of the receptive field (like your blur).
- Use a consistent border strategy (same as in Assignment 3) and return an output image (same dtype/range where applicable).

---

## 🧪 Real Filters

### Assignment 5a — Gaussian Blur (5×5)
**Code cell variable:** `gauss_filter`

**Goal:** Use a normalized 5×5 **Gaussian kernel** to blur.

**What to implement:**
- Define `gauss_filter` as:



- Convolve `gray_image` with `gauss_filter` using your `convolve`.
- Observe the smoother blur compared to the box filter.

---

### Assignment 5b — Sharpening
**Code cell variable:** `sharp_filter`

**Goal:** Enhance edges by amplifying local differences.

**What to implement:**
- Define the 3×3 sharpening kernel:


- Convolve `gray_image` with `sharp_filter`.
- Clip results to `[0,1]` after filtering.

---

### Assignment 6 — Edge Detection
**Code cell variable:** `sobel_magnitude`

**Goal:** Detect edges using gradient-like filters and magnitude.

**What to do:**
- Use provided edge kernels (e.g., Sobel-like / Laplacian shown in the text).
- Typical approach: convolve with **X** and **Y** edge filters to get `Ix` and `Iy`, then compute:


- Display the edge magnitude image (grayscale).

**Note:** Blurring beforehand reduces noise sensitivity.

---

## ⏬ Downsampling — Pooling

**Q1 (theory):** Explain why pooling helps control overfitting.  
(*Answer in the designated markdown cell.*)

### Assignment 7 — Max Pooling
**Function:** `def max_pool(image, kern_size, stride): -> pooled`

**Goal:** Implement 2D max pooling over a grayscale image.

**What to implement:**
- `kern_size` is a tuple `(kH, kW)`, `stride` is an integer.
- For each window, take the **max** value.
- Output dimensions:


- Return pooled image. Try different kernels/strides.

---

## 🧠 From Filters to CNNs

### Dense Baseline (Provided)
**Function (already implemented):**  
`build_neural_net(input_size, hidden_nodes, num_classes)`  
Builds a 784→hidden→softmax dense model for Fashion-MNIST.

---

### Assignment 8 — Feature Maps (Hand-coded CNN part 1)
**Function:** `def create_fmaps(filters, data): -> fmaps`

**Goal:** Produce multiple **feature maps** per image via convolution.

**What to implement:**
- `filters`: list of 2D kernels (e.g., `gauss_filter`, `sharp_filter`, `x_sobel_filter`, `y_sobel_filter`).
- `data`: array of images `(N, H, W)` (grayscale).
- For each filter *f* and each image, compute `convolve(image, f)` (use `fast_convolve(..., mode="same")` as permitted in the cell).
- Return a 4D array with shape `(num_filters, N, H, W)`.

---

### Assignment 9 — Pooling Features (Hand-coded CNN part 2)
**Function:** `def pool_fmaps(fmaps): -> fmaps_small`

**Goal:** Downsample each feature map with 2×2 max pooling.

**What to implement:**
- Input `fmaps` shape: `(num_filters, N, H, W)`.
- Output `fmaps_small` shape: `(num_filters, N, H//2, W//2)`  
(half each spatial dimension).
- Use `fast_pool(..., block_size=(1, 1, 2, 2), func=np.max)` or equivalent logic to pool per feature map.
- This becomes the input to a dense head after flattening.

---

### Assignment 10 — Keras CNN (Learned Features)
**Function:** `def build_conv_net(image_size, hidden_nodes, num_classes): -> model`

**Goal:** Replace hand-crafted filters with **learned** convolutional layers.

**What to implement:**
- Build a Keras `Sequential` model that starts with:
- `layers.Conv2D(...)` on inputs of shape `(image_size, image_size, 1)`
- `layers.MaxPool2D(...)`
- `layers.Flatten()`
- Dense classifier head (e.g., `Dense(hidden_nodes, activation='sigmoid')` then `Dense(num_classes, activation='softmax')`)
- Compile with `categorical_crossentropy`, track `accuracy`.
- Train and compare validation curves against the baseline dense net and hand-coded CNN.

---

## 🎯 Expected Outcomes

By the end you can:
- Implement **sepia**, **grayscale**, **box blur**, and general **convolution**.
- Apply real filters: **Gaussian**, **sharpen**, **edges**.
- Implement **max pooling** and compute output shapes.
- Build a dense classifier baseline and a **hand-coded CNN** (filters → feature maps → pooling → dense).
- Build and train a **Keras CNN**, and explain why learned filters often outperform fixed ones.

---

## 📝 Notes & Tips

- Keep all images in `[0, 1]`. Clip where needed.
- Document your **border strategy** (padding vs cropping).
- For Gaussian blur, use the provided normalized 5×5 kernel.
- For edge detection, compute gradient magnitude after convolving with X/Y kernels.
- Shapes matter: verify `(num_filters, N, H, W)` and halving after pooling.
