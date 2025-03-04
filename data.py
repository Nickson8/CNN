import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


# Load the ARFF file
def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df

# Plot the time series
def plot_eeg_data(df, sampling_rate=128):

    electrodes = [col for col in df.columns if col != 'eyeDetection']

    # # Identify changes in 'eyeState'
    # eye_state_changes = df['eyeDetection'].astype(float).diff().fillna(0).ne(0)
    # change_indices = df.index[eye_state_changes]

    # Create subplots
    num_electrodes = len(electrodes)
    fig, axes = plt.subplots(num_electrodes+1, 1, figsize=(15, 3 * (num_electrodes+1)), sharex=True)

    if num_electrodes == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for i, electrode in enumerate(electrodes):
        data = [x for x in df[electrode].astype(float)]
        # Generate time axis (assuming a uniform sampling rate)
        time_axis = [i / sampling_rate for i in range(len(data))]

        axes[i].plot(time_axis, data, label=electrode, linewidth=0.5)

        # if eye_state_changes is not None:
        #     for idx in change_indices:
        #         axes[i].axvline(x=idx / sampling_rate, color='red', linestyle='--', linewidth=0.5)

        #axes[i].set_title(f"EEG Time Series for Electrode: {electrode}")
        #axes[i].set_ylabel("Amplitude")
        axes[i].legend()
        axes[i].grid(True)

    row_sums = [x/14 for x in df.astype(float).sum(axis='columns')]
    # Generate time axis (assuming a uniform sampling rate)
    time_axis = [i / sampling_rate for i in range(len(row_sums))]

    axes[-1].plot(time_axis, row_sums, label="Media", linewidth=0.5)

    # if eye_state_changes is not None:
    #     for idx in change_indices:
    #         axes[-1].axvline(x=idx / sampling_rate, color='red', linestyle='--', linewidth=0.5)

    axes[-1].legend()
    axes[-1].grid(True)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    return fig

# Main function
def plot_eeg():
    file_path = "filtered_output.arff"
    try:
        df = load_arff(file_path)
        print("Columns in the dataset:", df.columns.tolist())

        # print(df.head())

        fig = plot_eeg_data(df)

        # Create a Tkinter window
        root = tk.Tk()
        root.title("EEG Data Viewer")

        # Add a scrollable canvas
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame)
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Embed the plot
        plot_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        root.mainloop()

    except Exception as e:
        print(f"An error occurred: {e}")