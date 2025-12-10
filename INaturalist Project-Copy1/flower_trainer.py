
import os
import numpy as np
import tensorflow as tf

def _find_flowering_idx(class_names, aliases=None):
    """
    Return the index in class_names that corresponds to 'flowering'.
    Tries common aliases and falls back for a 2-class setup where one is 'non-*'.
    """
    if aliases is None:
        aliases = {"flowering", "flower", "flowers"}

    lower = [c.strip().lower() for c in class_names]
    for i, name in enumerate(lower):
        if any(alias in name for alias in aliases):
            return i

    if len(lower) == 2:
        if "non" in lower[0]:
            return 1
        if "non" in lower[1]:
            return 0

    raise ValueError(
        f"Could not identify 'flowering' class. class_names={class_names}. "
        f"Pass clearer names or edit aliases."
    )

def percent_flowering(
    model,
    folder_path: str,
    img_height: int,
    img_width: int,
    class_names,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    return_counts: bool = False,
):
    """
    Compute the percentage of images in a folder predicted as the 'flowering' class.

    Parameters
    ----------
    model : tf.keras.Model
        Trained classification model.
    folder_path : str
        Path to a folder containing unlabeled images.
    img_height, img_width : int
        Size used during training (must match model input).
    class_names : list[str]
        Class label names in the same order used for training.
    extensions : tuple[str]
        Image file extensions to include.
    return_counts : bool
        If True, return (percentage, (flowering_count, total_count)).

    Returns
    -------
    float or (float, (int, int))
        Percentage flowering, optionally with raw counts.
    """
    flowering_idx = _find_flowering_idx(class_names)

    flowering_count = 0
    total_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            img_path = os.path.join(folder_path, filename)

            img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            preds = model.predict(img_array, verbose=0)     # (1, num_classes)
            probs = tf.nn.softmax(preds[0]).numpy()         # (num_classes,)
            pred_idx = int(np.argmax(probs))

            total_count += 1
            if pred_idx == flowering_idx:
                flowering_count += 1

    if total_count == 0:
        if return_counts:
            return 0.0, (0, 0)
        return 0.0

    pct = 100.0 * flowering_count / total_count
    if return_counts:
        return pct, (flowering_count, total_count)
    return pct





