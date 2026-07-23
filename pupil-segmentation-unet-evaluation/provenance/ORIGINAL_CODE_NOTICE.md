# Third-party source notice

The directory `original_author_code/` is an unmodified copy of the archive supplied from the public repository:

`naveen-purohit/Iris-Segmentation-through-deep-learning-UNet`

Original project author information remains inside the original files. The added evaluation pipeline does not alter those files and loads `create_model()` from the original `UNet_iris_segmentation.py` at runtime.

No separate license file was present in the supplied third-party archive. Therefore, the license for the added scripts does **not** apply to `original_author_code/`. Users are responsible for complying with the original author's terms and applicable copyright law.

Integrity can be checked with:

```bash
python tools/verify_original.py
```
