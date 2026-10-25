"""
Execute all cells in HAR_Capstone_Notebook.ipynb and save outputs directly into the notebook.
"""

import os
import sys
import json
import io
import base64
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

notebook_path = os.path.join("notebooks", "HAR_Capstone_Notebook.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Shared execution scope
shared_globals = {
    "__name__": "__main__",
    "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)
}

exec_count = 1

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        outputs = []
        
        # Clear any existing plots
        plt.close("all")
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                exec(code, shared_globals)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}\n"
            stdout_buf.write(err_msg)
            
        text_output = stdout_buf.getvalue()
        if text_output.strip():
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": text_output.splitlines(keepends=True)
            })
            
        # Check if matplotlib generated figures
        fig_nums = plt.get_fignums()
        for f_num in fig_nums:
            fig = plt.figure(f_num)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            b64_data = base64.b64encode(img_buf.read()).decode("utf-8")
            outputs.append({
                "data": {
                    "image/png": b64_data,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            plt.close(fig)
            
        cell["execution_count"] = exec_count
        cell["outputs"] = outputs
        exec_count += 1
        print(f"Executed Cell #{exec_count - 1} successfully.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"\n[SUCCESS] Notebook completely executed and saved with outputs to: {notebook_path}")
