from __future__ import annotations

import tkinter as tk
from tkinter import filedialog


def choose_video() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = ""
    while not path:
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")],
        )

    root.destroy()
    return path
