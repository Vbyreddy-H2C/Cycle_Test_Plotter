import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re
import numpy as np
import os
from datetime import datetime, timedelta
from fpdf import FPDF



# --- Data Processing Functions (Unchanged) ---
def parse_time_to_seconds(time_str):
    # This function remains the same
    if not re.match(r'^\d{1,2}:\d{1,2}(:\d{1,2}(\.\d+)?)?$', time_str):
        raise ValueError(f"Invalid time format: {time_str}")
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s_part = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s_part
    elif len(parts) == 2:
        h, m = int(parts[0]), int(parts[1])
        return h * 3600 + m * 60
    raise ValueError(f"Unexpected time format after regex match: {time_str}")


def process_file_content(file_path):
    # This function remains the same
    cleaned_data_rows = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read file: {e}")
        return None
    lines = raw_content.replace('\x00', '').splitlines()

    for line_content in lines:
        line_content = line_content.strip().replace('"', '').replace("'", "")
        if not line_content: continue
        parts = [p.strip() for p in re.split(r',|\t|\s+', line_content) if p.strip()]

        data_to_process = None
        if len(parts) >= 8:
            data_to_process = parts[1:8]
        elif len(parts) == 7:
            data_to_process = parts[0:7]

        if data_to_process:
            try:
                time_str, cycle, p_max, p_min, s_max, s_min, temp = data_to_process
                parse_time_to_seconds(time_str)
                cleaned_data_rows.append(
                    {'TimeStr': time_str, 'Cycle': float(cycle), 'Pmax': float(p_max), 'Pmin': float(p_min),
                     'Smax': float(s_max), 'Smin': float(s_min), 'Temp': float(temp)})
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping line due to data error: {e}. Line: '{line_content}'")
        elif line_content:
            print(f"Warning: Skipping line due to insufficient columns. Line: '{line_content}'")

    if not cleaned_data_rows: return None
    df = pd.DataFrame(cleaned_data_rows)
    df['TimeInSeconds'] = df['TimeStr'].apply(lambda x: parse_time_to_seconds(x) if isinstance(x, str) else pd.NA)
    df.dropna(subset=['TimeInSeconds'], inplace=True)
    if df.empty: return None
    first_time_sec = df['TimeInSeconds'].iloc[0]
    day_offset_sec, prev_time_sec, elapsed_seconds = 0, first_time_sec, []
    for current_time_sec in df['TimeInSeconds']:
        if current_time_sec < prev_time_sec and (prev_time_sec - current_time_sec) > (20 * 3600):
            day_offset_sec += 24 * 3600
        elapsed_seconds.append(current_time_sec + day_offset_sec - first_time_sec)
        prev_time_sec = current_time_sec
    df['ElapsedTime'] = [s / 3600.0 for s in elapsed_seconds]
    df['TimePerSample_sec'] = df['ElapsedTime'].diff() * 3600
    numeric_cols = ['Cycle', 'Pmax', 'Pmin', 'Smax', 'Smin', 'Temp']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['DeltaP'] = df['Pmax'] - df['Pmin']
    df['DeltaS'] = df['Smax'] - df['Smin']
    df.dropna(subset=numeric_cols + ['ElapsedTime'], inplace=True)
    if df.empty: return None
    df.reset_index(drop=True, inplace=True)
    df['Cycle_Row_Based'] = df.index + 1
    return df


# --- NEW: Report Editor Dialog Class ---
class ReportEditorDialog(tk.Toplevel):
    def __init__(self, parent, initial_data):
        super().__init__(parent)
        self.title("Edit Report Details")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()

        self.initial_data = initial_data
        self.result = None

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        self.entries = {}
        for i, (key, value) in enumerate(self.initial_data.items()):
            ttk.Label(main_frame, text=f"{key}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            entry_var = tk.StringVar(value=value)
            entry = ttk.Entry(main_frame, textvariable=entry_var)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=5)
            self.entries[key] = entry_var

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(self.initial_data), column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Generate", command=self.on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=10)

        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.wait_window(self)

    def on_ok(self):
        self.result = {key: var.get() for key, var in self.entries.items()}
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


# --- GUI Application ---
class CycleDataAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Unchanged __init__ properties...
        self.title("Cycle Data Analyzer")
        self.geometry("1000x850")
        self.df = None
        self.file_path = tk.StringVar()
        self.ax2 = None
        self.x_axis_options = {"Time (hours)": "ElapsedTime", "Cycle (by row)": "Cycle_Row_Based"}
        self.y_axis_options = {"Max Pressure (bar)": "Pmax", "Min Pressure (bar)": "Pmin", "Max Stroke (%)": "Smax",
                               "Min Stroke (%)": "Smin", "Temperature": "Temp", "Delta Pressure (bar)": "DeltaP",
                               "Delta Stroke (%)": "DeltaS", "Pressure (bar, Max & Min)": ["Pmax", "Pmin"],
                               "Stroke (%, Max & Min)": ["Smax", "Smin"], "Time per Sample (sec)": "TimePerSample_sec"}
        self.STROKE_DESCRIPTIVE_KEYS = {"Max Stroke (%)", "Min Stroke (%)", "Delta Stroke (%)", "Stroke (%, Max & Min)"}
        self.x_axis_var = tk.StringVar()
        self.plot_middle_segment_var = tk.BooleanVar(value=False)
        self.total_time_var = tk.StringVar(value="N/A")
        self.total_cycles_var = tk.StringVar(value="N/A")
        self.avg_time_cycle_var = tk.StringVar(value="N/A")
        self.avg_cycles_min_var = tk.StringVar(value="N/A")
        self.high_pressure_threshold_var = tk.StringVar(value="100.0")
        self.low_pressure_threshold_var = tk.StringVar(value="10.0")
        self.cycles_above_high_p_var = tk.StringVar(value="N/A")
        self.cycles_below_low_p_var = tk.StringVar(value="N/A")
        self._setup_ui()

    def _setup_ui(self):
        # --- MODIFICATION: Add "Generate Report" button ---
        # ... (all previous setup code up to plotting_controls_frame)
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)
        controls_main_frame = ttk.Frame(main_paned_window, padding="10", width=350)
        controls_main_frame.pack_propagate(False)
        main_paned_window.add(controls_main_frame, weight=1)
        file_frame = ttk.LabelFrame(controls_main_frame, text="File Operations", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_frame, text="Data File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path, state="readonly")
        self.file_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.load_file)
        self.browse_button.pack(side=tk.LEFT, padx=5)
        self.axis_selection_frame = ttk.LabelFrame(controls_main_frame, text="Axis Selection", padding="10")
        self.axis_selection_frame.pack(fill=tk.X, pady=5)
        self.axis_selection_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(self.axis_selection_frame, text="X-Axis:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.x_axis_dropdown = ttk.Combobox(self.axis_selection_frame, textvariable=self.x_axis_var,
                                            values=list(self.x_axis_options.keys()), state="readonly", width=25)
        self.x_axis_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        if self.x_axis_options: self.x_axis_dropdown.current(0)
        ttk.Label(self.axis_selection_frame, text="Y-Axis (select one or more):").grid(row=1, column=0, columnspan=2,
                                                                                       sticky=tk.W, padx=5,
                                                                                       pady=(10, 2))
        self.y_axis_listbox = tk.Listbox(self.axis_selection_frame, selectmode=tk.EXTENDED, exportselection=False,
                                         height=len(self.y_axis_options))
        for option in self.y_axis_options.keys(): self.y_axis_listbox.insert(tk.END, option)
        self.y_axis_listbox.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=2)
        y_scrollbar = ttk.Scrollbar(self.axis_selection_frame, orient=tk.VERTICAL, command=self.y_axis_listbox.yview)
        y_scrollbar.grid(row=2, column=2, sticky='ns')
        self.y_axis_listbox.config(yscrollcommand=y_scrollbar.set)

        plotting_controls_frame = ttk.LabelFrame(controls_main_frame, text="Plotting & Export",
                                                 padding="10")  # Changed label
        plotting_controls_frame.pack(fill=tk.X, pady=5)

        self.plot_button = ttk.Button(plotting_controls_frame, text="Plot Data", command=self.plot_data,
                                      state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_figure_button = ttk.Button(plotting_controls_frame, text="Save Figure", command=self._save_figure,
                                             state=tk.DISABLED)
        self.save_figure_button.pack(side=tk.LEFT, padx=5, pady=5)

        # New Button
        self.generate_report_button = ttk.Button(plotting_controls_frame, text="Generate Report",
                                                 command=self._generate_report, state=tk.DISABLED)
        self.generate_report_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.mid_segment_checkbox = ttk.Checkbutton(plotting_controls_frame, text="Plot Middle Segment",
                                                    variable=self.plot_middle_segment_var,
                                                    command=self._toggle_plotting_widgets)
        self.mid_segment_checkbox.pack(side='left', padx=10, pady=5, expand=True)  # expand to fill space
        # ... (rest of the UI setup is the same)
        metrics_frame = ttk.LabelFrame(controls_main_frame, text="Data Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=5, expand=True)
        metrics_frame.columnconfigure(1, weight=1)
        ttk.Label(metrics_frame, text="Total Time (hr):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.total_time_var, font="TkDefaultFont 10 bold").grid(row=0, column=1,
                                                                                                      sticky=tk.W,
                                                                                                      padx=5, pady=2)
        ttk.Label(metrics_frame, text="Total Cycles:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.total_cycles_var, font="TkDefaultFont 10 bold").grid(row=1, column=1,
                                                                                                        sticky=tk.W,
                                                                                                        padx=5, pady=2)
        ttk.Label(metrics_frame, text="Avg Time/Cycle (s):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.avg_time_cycle_var, font="TkDefaultFont 10 bold").grid(row=2,
                                                                                                          column=1,
                                                                                                          sticky=tk.W,
                                                                                                          padx=5,
                                                                                                          pady=2)
        ttk.Label(metrics_frame, text="Avg Cycles/Min:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.avg_cycles_min_var, font="TkDefaultFont 10 bold").grid(row=3,
                                                                                                          column=1,
                                                                                                          sticky=tk.W,
                                                                                                          padx=5,
                                                                                                          pady=2)
        sep = ttk.Separator(metrics_frame, orient=tk.HORIZONTAL)
        sep.grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(metrics_frame, text="High P Threshold (bar):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.high_p_entry = ttk.Entry(metrics_frame, textvariable=self.high_pressure_threshold_var, width=10,
                                      state=tk.DISABLED)
        self.high_p_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, text="Cycles > High P:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.cycles_above_high_p_var, font="TkDefaultFont 10 bold").grid(row=6,
                                                                                                               column=1,
                                                                                                               sticky=tk.W,
                                                                                                               padx=5,
                                                                                                               pady=2)
        ttk.Label(metrics_frame, text="Low P Threshold (bar):").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.low_p_entry = ttk.Entry(metrics_frame, textvariable=self.low_pressure_threshold_var, width=10,
                                     state=tk.DISABLED)
        self.low_p_entry.grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, text="Cycles < Low P:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.cycles_below_low_p_var, font="TkDefaultFont 10 bold").grid(row=8,
                                                                                                              column=1,
                                                                                                              sticky=tk.W,
                                                                                                              padx=5,
                                                                                                              pady=2)
        self.update_pressure_button = ttk.Button(metrics_frame, text="Update Pressure Counts",
                                                 command=self._update_pressure_counts, state=tk.DISABLED)
        self.update_pressure_button.grid(row=9, column=0, columnspan=2, pady=(10, 0))
        plot_main_frame = ttk.Frame(main_paned_window, padding="5")
        main_paned_window.add(plot_main_frame, weight=3)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_main_frame)
        self.toolbar.update()
        self.status_var = tk.StringVar(value="Ready. Please load a data file.")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ... (existing methods _toggle_plotting_widgets, _save_figure are unchanged) ...
    def _toggle_plotting_widgets(self):
        if self.plot_middle_segment_var.get():
            self.x_axis_dropdown.config(state=tk.DISABLED)
            self.y_axis_listbox.config(state=tk.DISABLED)
            self.status_var.set("Plot mode set to 'Middle Segment'.")
        else:
            self.x_axis_dropdown.config(state="readonly")
            if self.df is not None and not self.df.empty: self.y_axis_listbox.config(state=tk.NORMAL)
            self.status_var.set("Plot mode set to 'Standard'.")

    def _save_figure(self):
        if not (self.ax.lines or (self.ax2 and self.ax2.lines)):
            messagebox.showwarning("Save Error", "No plot available to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"),
                                                           ("SVG", "*.svg")], title="Save Figure")
        if not filepath: return
        try:
            self.figure.savefig(filepath, dpi=300)
            messagebox.showinfo("Save Successful", f"Figure saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save figure:\n{e}")

    def load_file(self):
        filepath = filedialog.askopenfilename(title="Select Data File", filetypes=(
            ("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")))
        if not filepath: return
        self.file_path.set(filepath)
        self.status_var.set(f"Processing {filepath}...")
        self.update_idletasks()
        self.df = process_file_content(filepath)
        if self.df is not None and not self.df.empty:
            # --- MODIFICATION: Enable report button on load ---
            for widget in [self.plot_button, self.high_p_entry, self.low_p_entry, self.update_pressure_button,
                           self.generate_report_button]:
                widget.config(state=tk.NORMAL)
            # --- END MODIFICATION ---
            self._toggle_plotting_widgets()
            if self.y_axis_listbox.size() > 0:
                self.y_axis_listbox.selection_clear(0, tk.END)
                self.y_axis_listbox.selection_set(0)
            self.status_var.set(f"File loaded. Shape: {self.df.shape}. Ready to plot.")
            self._update_and_display_metrics()
            self.plot_data()
        else:
            # --- MODIFICATION: Disable report button on failure ---
            for widget in [self.plot_button, self.save_figure_button, self.high_p_entry, self.low_p_entry,
                           self.update_pressure_button, self.generate_report_button]:
                widget.config(state=tk.DISABLED)
            # --- END MODIFICATION ---
            self.ax.clear()
            if self.ax2: self.ax2.remove(); self.ax2 = None
            self.ax.text(0.5, 0.5, "Failed to load/process data.", ha='center', va='center')
            self.canvas.draw_idle()
            self._update_and_display_metrics(clear=True)
            self.status_var.set("Failed to load or process. Check console.")
            messagebox.showerror("Error", "No valid data processed. Check file and console.")

    # ... (all other _update and plot methods are unchanged) ...
    def _update_and_display_metrics(self, clear=False):
        if clear or self.df is None or self.df.empty:
            for var in [self.total_time_var, self.total_cycles_var, self.avg_time_cycle_var, self.avg_cycles_min_var]:
                var.set("N/A")
            self._update_pressure_counts(clear=True)
            return
        try:
            total_time_hr = self.df['ElapsedTime'].max()
            total_cycles = len(self.df)

            self.total_time_var.set(f"{total_time_hr:.2f}")
            self.total_cycles_var.set(f"{total_cycles:,.0f}")
            if total_cycles > 0 and total_time_hr > 0:
                self.avg_time_cycle_var.set(f"{(total_time_hr * 3600) / total_cycles:.2f}")
                self.avg_cycles_min_var.set(f"{total_cycles / (total_time_hr * 60):.2f}")
            else:
                self.avg_time_cycle_var.set("N/A")
                self.avg_cycles_min_var.set("N/A")
            self._update_pressure_counts()
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self._update_and_display_metrics(clear=True)

    def _update_pressure_counts(self, clear=False):
        if clear or self.df is None or self.df.empty:
            self.cycles_above_high_p_var.set("N/A")
            self.cycles_below_low_p_var.set("N/A")
            return
        try:
            high_p, low_p = float(self.high_pressure_threshold_var.get()), float(self.low_pressure_threshold_var.get())
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Input", "Pressure thresholds must be numeric values.")
            return

        cycles_above = len(self.df[self.df['Pmax'] > high_p])
        cycles_below = len(self.df[self.df['Pmin'] < low_p])
        self.cycles_above_high_p_var.set(f"{cycles_above:,}")
        self.cycles_below_low_p_var.set(f"{cycles_below:,}")

        self.status_var.set("Pressure cycle counts updated.")

    def plot_data(self):
        if self.df is None or self.df.empty:
            messagebox.showerror("Error", "No data loaded.")
            return
        if self.plot_middle_segment_var.get():
            self._plot_middle_segment()
        else:
            self._plot_standard()

    def _apply_custom_grid(self, axis_obj):
        """Applies a standard, simple grid to the axis."""
        if axis_obj:
            axis_obj.grid(True, linestyle='--', linewidth='0.5', color='gray')

    def _plot_middle_segment(self):
        self.status_var.set("Generating middle segment plot...")
        mid_point_hr = self.df['ElapsedTime'].max() / 2
        start_hr = mid_point_hr - (30 / 3600.0)
        end_hr = mid_point_hr + (30 / 3600.0)
        segment_df = self.df[(self.df['ElapsedTime'] >= start_hr) & (self.df['ElapsedTime'] <= end_hr)].copy()

        required_points = 10
        if len(segment_df) < required_points:
            try:
                start_index = self.df[self.df['ElapsedTime'] >= start_hr].index[0]
                end_index_for_segment = start_index + required_points
                if end_index_for_segment > len(self.df):
                    end_index_for_segment = len(self.df)
                segment_df = self.df.iloc[start_index: end_index_for_segment].copy()
            except IndexError:
                # Fallback if no points are in the window, just take the middle 10 points of the whole dataset
                if len(self.df) > required_points:
                    mid_idx = len(self.df) // 2
                    segment_df = self.df.iloc[mid_idx - (required_points // 2): mid_idx + (required_points // 2)].copy()
                else:
                    segment_df = self.df.copy()

        if segment_df.empty or len(segment_df) < 2:
            messagebox.showwarning("Plot Error", "Not enough data for a meaningful plot.")
            self.status_var.set("Plot Error.")
            return

        segment_df['SegmentTime_s'] = (segment_df['ElapsedTime'] - segment_df['ElapsedTime'].iloc[0]) * 3600

        time_points = segment_df['SegmentTime_s'].to_numpy()
        p_min_vals = segment_df['Pmin'].to_numpy()
        p_max_vals = segment_df['Pmax'].to_numpy()
        plot_x = np.repeat(time_points, 2)
        plot_y = np.column_stack((p_min_vals, p_max_vals)).ravel()

        self.ax.clear()
        if self.ax2: self.ax2.remove(); self.ax2 = None
        self.ax.plot(plot_x, plot_y, label='Pressure Range (Min-Max)')
        self.ax.set_title("Middle Segment Pressure Profile")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Pressure (bar)")
        self.ax.legend(loc='best')
        self._apply_custom_grid(self.ax)
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.save_figure_button.config(state=tk.NORMAL)
        self.status_var.set("Middle segment plot updated.")

    def _plot_standard(self):
        selected_x_key, selected_y_indices = self.x_axis_var.get(), self.y_axis_listbox.curselection()
        if not selected_x_key or not selected_y_indices:
            messagebox.showerror("Error", "Please select X and Y axes.")
            return
        x_col = self.x_axis_options.get(selected_x_key)
        if x_col not in self.df.columns:
            messagebox.showerror("Plot Error", f"X-axis data ('{x_col}') is missing.")
            return

        self.ax.clear()
        if self.ax2: self.ax2.remove(); self.ax2 = None
        plot_df = self.df.copy()
        plotting_tasks, seen_cols = [], set()
        non_stroke_keys, stroke_keys = set(), set()
        for index in selected_y_indices:
            key = list(self.y_axis_options.keys())[index]
            y_spec = self.y_axis_options.get(key)
            is_stroke = key in self.STROKE_DESCRIPTIVE_KEYS
            if is_stroke:
                stroke_keys.add(key)
            else:
                non_stroke_keys.add(key)
            cols = y_spec if isinstance(y_spec, list) else [y_spec]
            for col in cols:
                if col not in plot_df.columns or plot_df[col].isnull().all():
                    messagebox.showerror("Plot Error", f"Y-axis data ('{col}' for '{key}') missing.")
                    return
                if col not in seen_cols:
                    plotting_tasks.append({'col': col, 'label': key, 'is_stroke': is_stroke})
                    seen_cols.add(col)
        if not plotting_tasks:
            messagebox.showinfo("Plot Info", "No unique Y-axis columns to plot.")
            return
        plot_df.dropna(subset=[x_col] + [t['col'] for t in plotting_tasks], inplace=True)
        if plot_df.empty:
            self.ax.text(0.5, 0.5, "No valid data for selection.", ha='center')
            self.canvas.draw_idle()
            return
        is_mixed_plot = bool(non_stroke_keys) and bool(stroke_keys)
        if is_mixed_plot: self.ax2 = self.ax.twinx()
        for task in plotting_tasks:
            target_ax = self.ax2 if is_mixed_plot and task['is_stroke'] else self.ax
            target_ax.plot(plot_df[x_col], plot_df[task['col']], label=task['label'])
        self.ax.set_xlabel(selected_x_key)
        if is_mixed_plot:
            self.ax2.set_ylabel("Stroke (%)")
            is_all_pressure = bool(non_stroke_keys) and all('Pressure' in k for k in non_stroke_keys)
            self.ax.set_ylabel("Pressure (bar)" if is_all_pressure else " / ".join(
                sorted([k.split('(')[0].strip() for k in non_stroke_keys])))
        elif bool(stroke_keys):
            self.ax.set_ylabel("Stroke (%)")
        elif bool(non_stroke_keys):
            is_all_pressure = all('Pressure' in k for k in non_stroke_keys)
            self.ax.set_ylabel("Pressure (bar)" if is_all_pressure else " / ".join(
                sorted([k.split('(')[0].strip() for k in non_stroke_keys])))
        self.ax.set_title(f"Plot vs. {selected_x_key}")
        lines, labels = self.ax.get_legend_handles_labels()
        if self.ax2:
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            lines.extend(lines2)
            labels.extend(labels2)
        if lines: self.ax.legend(lines, labels, loc='best')
        self._apply_custom_grid(self.ax)
        if self.ax2: self._apply_custom_grid(self.ax2)
        self.ax.set_xlim(left=0)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.save_figure_button.config(state=tk.NORMAL)
        self.status_var.set("Plot updated.")

    # --- NEW: Report Generation Method ---
    def _generate_report(self):
        if self.df is None or self.df.empty:
            messagebox.showerror("Error", "No data available to generate a report.")
            return

        self.status_var.set("Gathering data for report...")
        self.update_idletasks()

        # 1. Gather data for the editor dialog
        total_hours = self.df['ElapsedTime'].max()
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(hours=total_hours)

        initial_data = {
            "Test Name": os.path.splitext(os.path.basename(self.file_path.get()))[0],
            "Test Standard": "EN 17339:2020 -T6",
            "Test Date": end_datetime.strftime('%d-%m-%Y'),
            "Test Start Time": start_datetime.strftime('%d-%m-%Y %H:%M:%S'),
            "Test End Time": end_datetime.strftime('%d-%m-%Y %H:%M:%S'),
            "Number of Cycles": self.total_cycles_var.get(),
            f"Cycles Above {self.high_pressure_threshold_var.get()} bar": self.cycles_above_high_p_var.get(),
            f"Cycles Below {self.low_pressure_threshold_var.get()} bar": self.cycles_below_low_p_var.get(),
            "Max Temperature (°C)": f"{self.df['Temp'].max():.2f}",
            "Average Cycles per Minute": self.avg_cycles_min_var.get(),
            "Average Time per Cycle (s)": self.avg_time_cycle_var.get()
        }

        # 2. Show the editor dialog
        dialog = ReportEditorDialog(self, initial_data)
        final_data = dialog.result

        if not final_data:
            self.status_var.set("Report generation cancelled.")
            return

        # 3. Prompt for save location
        base_name = final_data["Test Name"]
        save_path_base = os.path.splitext(filedialog.asksaveasfilename(
            initialfile=f"{base_name}_Report.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf")],
            title="Save Report As"
        ))[0]

        if not save_path_base:
            self.status_var.set("Report generation cancelled.")
            return

        pdf_path = save_path_base + ".pdf"
        png_path = save_path_base + ".png"

        self.status_var.set("Generating report... this may take a moment.")
        self.update_idletasks()

        try:
            # 4. Generate a single figure containing the graph and table
            report_fig = Figure(figsize=(8.27, 11.69), dpi=150)
            gs = plt.GridSpec(2, 1, height_ratios=[2.5, 1], figure=report_fig)

            # --- Top part: The Graph (MODIFIED) ---
            ax1 = report_fig.add_subplot(gs[0])
            ax1.plot(self.df['Cycle_Row_Based'], self.df['Pmax'], label='Max Pressure', color='blue', linewidth=0.8)
            ax1.plot(self.df['Cycle_Row_Based'], self.df['Pmin'], label='Min Pressure', color='cyan', linewidth=0.8)
            ax1.set_xlabel('Number of Cycles')  # Changed from 'Time (hours)'
            ax1.set_ylabel('Pressure (bar)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, linestyle='--', linewidth='0.5', color='gray')
            ax1.set_xlim(left=0)

            ax2 = ax1.twinx()
            ax2.plot(self.df['Cycle_Row_Based'], self.df['Temp'], label='Temperature', color='red',
                     linewidth=1.0)  # Changed from ElapsedTime
            ax2.set_ylabel('Temperature (°C)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Combine legends from both axes and place it below the plot
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

            # --- Bottom part: The Table ---
            table_ax = report_fig.add_subplot(gs[1])
            table_ax.axis('off')

            table_data = [[f' {key}', f' {value}'] for key, value in final_data.items()]
            col_widths = [0.45, 0.45]

            the_table = table_ax.table(cellText=table_data, colWidths=col_widths, loc='center', cellLoc='left')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 1.8)

            report_fig.tight_layout(pad=3.0)

            # 5. Save the composite figure as a PNG
            report_fig.savefig(png_path)
            plt.close(report_fig)

            # 6. Generate the PDF and place the saved PNG inside it
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()
            pdf.image(png_path, x=10, y=10, w=190)
            pdf.output(pdf_path)

            messagebox.showinfo("Success",
                                f"Report files saved successfully!\n\nPDF: {pdf_path}\nPNG: {png_path}")
            self.status_var.set("Report generated successfully.")

        except Exception as e:
            messagebox.showerror("Report Error", f"An error occurred while generating the report:\n{e}")
            self.status_var.set("Report generation failed.")


if __name__ == '__main__':
    app = CycleDataAnalyzerApp()
    app.mainloop()