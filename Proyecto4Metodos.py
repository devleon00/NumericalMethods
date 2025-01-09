import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from mpl_interactions import ioff, panhandler, zoom_factory
from abc import ABC, abstractmethod
from tkinter import messagebox
from PIL import Image, ImageTk
import sympy as sy 
import math

def zoom_factory(ax, base_scale=1.2):
    """
    Attach a zooming functionality to a matplotlib Axes object that focuses on the mouse position.

    Parameters:
    - ax: The matplotlib Axes object.
    - base_scale: The zoom scaling factor. Default is 1.2 for slower zooming.

    Returns:
    - Disconnect function to disable the zoom functionality.
    """
    def zoom(event):
        if event.inaxes != ax:
            return

        # Get the current x and y limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Get the mouse position in data coordinates
        x_mouse = event.xdata
        y_mouse = event.ydata

        # Determine the scale factor
        if event.button == 'up':  # Scroll up to zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # Scroll down to zoom out
            scale_factor = base_scale
        else:
            return

        # Calculate new limits based on the mouse position
        new_x_min = x_mouse - (x_mouse - x_min) * scale_factor
        new_x_max = x_mouse + (x_max - x_mouse) * scale_factor
        new_y_min = y_mouse - (y_mouse - y_min) * scale_factor
        new_y_max = y_mouse + (y_max - y_mouse) * scale_factor

        # Apply the new limits
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)

        # Redraw the canvas
        ax.figure.canvas.draw()

    # Connect the zoom function to scroll events
    fig = ax.get_figure()
    cid = fig.canvas.mpl_connect('scroll_event', zoom)

    # Return a disconnect function to disable zoom
    return lambda: fig.canvas.mpl_disconnect(cid)

class GraphPlotter:
    def __init__(self, root, function):
        """
        Initializes the GraphPlotter object.

        Parameters:
        - root: The Tkinter root window.
        - function: The mathematical function to be plotted.
        """
        self.root = root
        self.function = function
        self.figsize = (8, 6)  # Store figsize if needed later

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

        # Create the canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Enable zoom and pan
        self.disconnect_zoom = zoom_factory(self.ax)
        self.pan_handler = panhandler(self.fig)
   
        # Initialize the plot
        self.initialize_plot()

    def initialize_plot(self):
        """Initializes or resets the plot components."""
        self.function_str = "1 / x - 1"
        self.function = eval(f"lambda x: {self.function_str}")
        self.a = 0.25
        self.b = 2
        self.tolerance = 0.01  # Default tolerance value
        self.plot = []

        # Clear the axes and set limits
        self.ax.clear()
        # self.x_range = ()
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlim(-3, 3)

        # Update the plot with initial settings
        self.update()

    def reset_graph(self):
        """Resets the graph to its initial state."""
        self.initialize_plot()
        
    def setup_grid(self):
        """Sets up the grid lines for the plot."""
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)

        self.ax.set_title(f'Graph of f(x) = {self.function_str}')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')

    def _setup_plot(self):
        """Sets up the initial plot with axis labels, title, and grid."""
        self.setup_grid()
        
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
        self.ax.axvline(x=self.a, color='gray', linestyle='--', label=f'a_initial = {self.a}')
        self.ax.axvline(x=self.b, color='gray', linestyle='--', label=f'b_initial = {self.b}')

        self.plot_function()
        
        plt.legend(loc='upper left')

    def plot_function(self):
        """Plots the function over the current x_range."""
        if self.plot:
            try:
                self.plot[0].remove()
            except:
                pass

        self.x_range = self.ax.get_xlim()
        self.x_values = np.linspace(self.x_range[0], self.x_range[1], 1000)
        self.y_values = self.function(self.x_values)
        
        self.plot = self.ax.plot(self.x_values, self.y_values, color='blue', label='f(x) = ' + self.function_str)

        self.canvas.draw()
        
    def clear_plot(self):
        """Clears the plot."""
        y_limits = self.ax.get_ylim()
        x_limits = self.ax.get_xlim()
        
        self.ax.clear()
        
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)

    def update(self):
        y_limits = self.ax.get_ylim()
        x_limits = self.ax.get_xlim()
        
        self.ax.clear()
        self._setup_plot()
        
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        self.plot_function()

    def plot_tolerance_area(self, ax, x_values, y_values, not_in_interval=False):
        """Plots the area where y-values are within the tolerance of y = 0."""
        # If not in interval is true then the tolerance area will be plotted outside the interval
        if not_in_interval:
            within_tolerance = (np.abs(y_values) <= self.tolerance)
        else:
            within_tolerance = (np.abs(y_values) <= self.tolerance) & (x_values >= self.a) & (x_values <= self.b)
        
        indices = np.where(within_tolerance)[0]
        
        if indices.size > 0:
            x_min = x_values[indices[0]]
            x_max = x_values[indices[-1]]
            ax.axvspan(x_min, x_max, color='yellow', alpha=0.3, label='Tolerance Area')
            
    def fill_areas(self, method_a, method_b):
        """Rellena las áreas inicial y actual en la gráfica."""
        print("Rellenando áreas")
        x_fill_initial = np.linspace(self.a, self.b, 1000)
        y_fill_initial = self.function(x_fill_initial)
        self.ax.fill_between(x_fill_initial, y_fill_initial, color='gray', alpha=0.3, label='Área inicial')

        x_fill_actual = np.linspace(method_a, method_b, 1000)
        y_fill_actual = self.function(x_fill_actual)
        self.ax.fill_between(x_fill_actual, y_fill_actual, color='lightblue', alpha=0.5, label='Área actual')

class NumericMethod(ABC):
    @abstractmethod
    def step(self):
        """Performs a single iteration step for the method."""
        pass

    @abstractmethod
    def plot_step(self):
        """Plots the current step of the method using the given GraphPlotter."""
        pass
    
    @abstractmethod
    def create_iteration_window(self):
        """Creates a new window to perform iterations of the method."""
        pass
    
    @abstractmethod
    def next_iteration(self):
        """Performs the next iteration of the method and updates the plot."""
        pass
    
class BisectionMethod(NumericMethod):
    def __init__(self, graph_plotter):
        self.plotter = graph_plotter

        self.a = self.plotter.a
        self.b = self.plotter.b
        self.m = None
        self.function = self.plotter.function
        self.tolerance = self.plotter.tolerance
        self.attempt = 0
        self.iteration_data = []
        self.create_iteration_window()
        

    def step(self):
        self.m = (self.a + self.b) / 2
        
        f_m = self.function(self.m)
        
        self.attempt += 1
        
        error =  (self.plotter.b - self.plotter.a) / (2 ** self.attempt)
        
        self.iteration_data.append((self.attempt, self.a, self.b, self.m, error))
        
        # Plot the current step
        self.plot_step()
        
        if np.abs(f_m) < self.tolerance:
            self.next_button.pack_forget()  # Hide the button when the root is found
            return self.m  # Root found
        
        if np.sign(f_m) == np.sign(self.function(self.a)):
            self.a = self.m
        else:
            self.b = self.m
        

    def plot_step(self):
        """Plots the current step of the bisection method using GraphPlotter."""
        self.plotter.update()

        # Plot the interval and midpoint
        self.plotter.ax.axvline(x=self.a, color='red', linestyle='--', label=f'a = {self.a}')
        self.plotter.ax.axvline(x=self.b, color='green', linestyle='--',  label=f'b = {self.b}')
        self.plotter.ax.axvline(x=self.m, color='purple', linestyle='--', label=f'm = {self.m}')

        self.plotter.ax.text(self.a, 3, 'A', fontsize=12, color='red', ha='center')
        self.plotter.ax.text(self.m, 1.5, 'M', fontsize=12, color='purple', ha='center')
        self.plotter.ax.text(self.b, 0.5, 'B', fontsize=12, color='green', ha='center')

        self.plotter.ax.text(self.a, -1, f'{self.a}', fontsize=12, color='red', ha='center')
        self.plotter.ax.text(self.m, -1.5, f'{self.m}', fontsize=12, color='purple', ha='center')
        self.plotter.ax.text(self.b, -2, f'{self.b}', fontsize=12, color='green', ha='center')

        m_y = self.function(self.m)
        self.plotter.ax.scatter(self.m, m_y, color='purple', zorder=5)
        self.plotter.ax.text(self.m, m_y, f'({self.m:.3f}, {m_y:.3f})', fontsize=10, color='purple', ha='right')

        # Plot the tolerance area
        self.plotter.plot_tolerance_area(self.plotter.ax, self.plotter.x_values, self.plotter.y_values)
        
        # Plot fill areas
        self.plotter.fill_areas(self.a, self.b)

        # self.plotter.plot_tolerance_area(plotter.ax, plotter.x_values, plotter.y_values)
        plt.legend(loc=(0,0))

        self.plotter.ax.legend()
        self.plotter.canvas.draw()

    def create_iteration_window(self):
        """Creates a new window to perform iterations of the bisection method."""
        # Create a top-level window with a small initial size
        iteration_window = tk.Toplevel(self.plotter.root)
        iteration_window.title("Bisection Method Iteration")
        iteration_window.geometry("500x400")
        iteration_window.configure(bg="#f0f0f0")
        
        # Keep window on top of the main window
        iteration_window.attributes("-topmost", True)

            # Title label for the window, centered and styled
        title_label = tk.Label(iteration_window, text="Bisection Method Iteration", font=("Helvetica", 14, "bold"), bg="#f0f0f0", anchor="center")
        title_label.pack(fill=tk.X, pady=(10, 5))

        # Label to display the current iteration details
        self.iteration_label = tk.Label(iteration_window, text="Current iteration details", font=("Helvetica", 12), bg="#f0f0f0", anchor="center")
        self.iteration_label.pack(fill=tk.X, pady=(5, 10))

        # Add a button for the next iteration
        self.next_button = tk.Button(iteration_window, text="Next Iteration", command=self.next_iteration, font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", activebackground="#005f99", activeforeground="white", relief="raised")
        self.next_button.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Frame to hold iteration details
        iteration_frame = tk.Frame(iteration_window, bg="#f0f0f0")
        iteration_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a table-like structure for displaying iteration details
        headers = ["Attempt", "a", "b", "m (Midpoint)", "Error"]
        for col, header in enumerate(headers):
            label = tk.Label(iteration_frame, text=header, font=("Helvetica", 12, "bold"), bg="#d3d3d3", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid")
            label.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            iteration_frame.grid_columnconfigure(col, weight=1, uniform="col")  # Set equal weight and uniform group

        self.iteration_table = tk.Frame(iteration_frame, bg="#f0f0f0")
        self.iteration_table.grid(row=1, column=0, columnspan=len(headers), sticky="nsew")
        iteration_frame.grid_rowconfigure(1, weight=1)

        self.next_iteration()

    def update_iteration_table(self):
        """Updates the table with the latest iteration details."""
        for widget in self.iteration_table.winfo_children():
            widget.destroy()

        for i, (attempt, a, b, m, error) in enumerate(self.iteration_data):
            tk.Label(self.iteration_table, text=str(attempt), font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{a:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{b:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{m:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=3, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{error * 100:.4f}%", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=4, sticky="nsew", padx=1, pady=1)

        # Configure columns to have equal weight and uniform group
        for col in range(5):
            self.iteration_table.grid_columnconfigure(col, weight=1, uniform="col")

    def next_iteration(self):
        """Performs the next iteration of the bisection method and updates the plot."""
        is_root_found = self.step()
        
        if is_root_found:
            self.iteration_label.config(text="Root found at x = {:.4f}".format(self.m))
        else:
            print(self.iteration_data)
            self.iteration_label.config(text=f"Attempt: {self.iteration_data[-1][0]} | Approximation: {self.iteration_data[-1][3]:.4f} | Error: {self.iteration_data[-1][4]* 100:.4f}%")
            
        self.update_iteration_table()

class NewtonMethod(NumericMethod):
    def __init__(self, graph_plotter):
        """
        Initialize the NewtonMethod with a graph plotter.
        """
        # Initialize the NewtonMethod with the graph plotter
        self.plotter = graph_plotter
        self.function = self.plotter.function
        self.tolerance = self.plotter.tolerance
        self.x0 = self.plotter.a
        self.attempt = 0
        self.steps = []
        self.iteration_data = []
        
        # Create a symbolic expression and its derivative
        self.x = sy.Symbol('x')
        self.expr = sy.sympify(self.plotter.function_str)
        self.expr_diff = eval(f"lambda x: {sy.diff(self.expr, self.x)}")
 
        self.create_iteration_window()

    def step(self):
        """
        Perform one iteration of the Newton-Raphson method.
        """
        self.steps.append(self.x0)
        
        f_x0 = self.function(self.x0)
        
        self.attempt += 1
        
        x_new = self.x0 - f_x0 / self.expr_diff(self.x0)
        
        error = np.abs(x_new - self.x0)    
        
        self.x0 = self.x0 - f_x0 / self.expr_diff(self.x0)
        
        self.iteration_data.append((self.attempt, self.x0, f_x0, error))
        
        self.plot_step(self.plotter)

        if np.abs(f_x0) < self.tolerance:
            self.next_button.pack_forget()  # Hide the button when the root is found
            return self.x0  # Root found
        
    def plot_step(self, plotter):
        """Plots the current step of the Newton method using GraphPlotter."""
        self.plotter.update()

        if self.steps:
            for step in self.steps:
                x_tangent = np.linspace(step - 0.5, step + 0.5, 400)
                y_tangent = self.function(step) + self.expr_diff(step) * (x_tangent - step)
                self.plotter.ax.plot(x_tangent, y_tangent, color='gray', linestyle='--', linewidth=1)
                self.plotter.ax.scatter(step, self.function(step), color='red', zorder=20)
                self.plotter.ax.plot([step, step], [0, self.function(step)], color='red', linestyle='--')

        self.plot_tolerance_area(plotter.ax, plotter.x_values, plotter.y_values)

        plotter.ax.legend()
        plotter.canvas.draw()

    def create_iteration_window(self):
        """Creates a new window to perform iterations of the Newton method."""
        # Create a top-level window for iteration
        iteration_window = tk.Toplevel(self.plotter.root)
        iteration_window.title("Newton Method Iteration")
        iteration_window.geometry("500x400")
        iteration_window.configure(bg="#f0f0f0")

        # Keep window on top of the main window
        iteration_window.attributes("-topmost", True)

        # Title label for the window
        title_label = tk.Label(iteration_window, text="Newton Method Iteration", font=("Helvetica", 14, "bold"), bg="#f0f0f0", anchor="center")
        title_label.pack(fill=tk.X, pady=(10, 5))

        # Frame for x0 input and submit button
        x0_frame = tk.Frame(iteration_window, bg="#f0f0f0")
        x0_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Label to prompt user for initial value of x0
        x0_label = tk.Label(x0_frame, text="Enter the x0 value: ", font=("Helvetica", 12), bg="#f0f0f0")
        x0_label.pack(side=tk.LEFT)

        # Entry widget for x0 input
        self.entry_x0 = tk.Entry(x0_frame, font=("Helvetica", 12))
        self.entry_x0.pack(side=tk.LEFT, padx=(5, 10))

        # Button to submit the initial value of x0
        self.submit_x0_button = tk.Button(x0_frame, text="Submit x0", command=lambda: self.submit_x0(iteration_window), font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", activebackground="#005f99", activeforeground="white", relief="raised")
        self.submit_x0_button.pack(side=tk.LEFT)

        # Frame to hold iteration details (initially hidden)
        self.iteration_frame = tk.Frame(iteration_window, bg="#f0f0f0")

        # Add a button for the next iteration (initially hidden)
        self.next_button = tk.Button(self.iteration_frame, text="Next Iteration", command=self.next_iteration, font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", activebackground="#005f99", activeforeground="white", relief="raised")

        # Create a table-like structure for displaying iteration details (initially hidden)
        headers = ["Attempt", "x0", "f_x0", "Error"]
        for col, header in enumerate(headers):
            label = tk.Label(self.iteration_frame, text=header, font=("Helvetica", 12, "bold"), bg="#d3d3d3", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid")
            label.grid(row=1, column=col, sticky="nsew", padx=1, pady=1)
            self.iteration_frame.grid_columnconfigure(col, weight=1, uniform="col")  # Set equal weight and uniform group

        self.iteration_table = tk.Frame(self.iteration_frame, bg="#f0f0f0")
        self.iteration_table.grid(row=2, column=0, columnspan=len(headers), sticky="nsew")
        self.iteration_frame.grid_rowconfigure(2, weight=1)


    def submit_x0(self, iteration_window):
        """Handles the submission of the initial value for x0."""
        try:
            # Get the value from the entry widget and convert it to float
            self.x0 = float(self.entry_x0.get())
            self.iteration_label = tk.Label(iteration_window, text=f"Initial value set: x0 = {self.x0:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center")
            self.iteration_label.pack(fill=tk.X, pady=(5, 5))

            # Hide the x0 entry, label, and submit button
            self.entry_x0.pack_forget()
            self.submit_x0_button.pack_forget()
            iteration_window.nametowidget(self.entry_x0.master).pack_forget()

            # Show the iteration frame and next iteration button
            self.iteration_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.next_button.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=(5, 10))

            # Start with the first iteration
            self.next_iteration()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for x0.")

    def update_iteration_table(self):
        """Updates the table with the latest iteration details."""
        for widget in self.iteration_table.winfo_children():
            widget.destroy()

        for i, (attempt, x0, f_x0, error) in enumerate(self.iteration_data):
            tk.Label(self.iteration_table, text=str(attempt), font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{x0:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{f_x0:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{error * 100:.4f}%", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=3, sticky="nsew", padx=1, pady=1)

        # Configure columns to have equal weight and uniform group
        for col in range(4):
            self.iteration_table.grid_columnconfigure(col, weight=1, uniform="col")

    def next_iteration(self):
        """Performs the next iteration of the Newton method and updates the plot."""
        is_root_found = self.step()

        if is_root_found:
            self.iteration_label.config(text="Root found at x = {:.4f}".format(self.x0))
            self.next_button.grid_forget()  # Hide the next iteration button when the root is found
        else:
            self.iteration_label.config(text=f"Attempt: {self.iteration_data[-1][0]} | Approximation: {self.iteration_data[-1][1]:.4f} | Error: {self.iteration_data[-1][3]* 100:.4f}%")

        self.update_iteration_table()


    def plot_tolerance_area(self, ax, x_values, y_values):
        """Plots the vertical lines at x-values where the y-values are within the tolerance of y = 0."""
        # Define tolerance bounds for y
        y_tolerance_upper = self.tolerance
        y_tolerance_lower = -self.tolerance

        # Find the x-values where the y-values are within the tolerance range
        tolerance_indices = np.where((y_values >= y_tolerance_lower) & (y_values <= y_tolerance_upper))[0]

        # If there are values within the tolerance range, plot the tolerance area
        if len(tolerance_indices) > 0:
            # Filter values to be within the current interval [a, b]
            tolerance_indices = tolerance_indices[(x_values[tolerance_indices] >= self.plotter.a) & (x_values[tolerance_indices] <= self.plotter.b)]
            if len(tolerance_indices) > 0:
                x_min = x_values[tolerance_indices.min()]
                x_max = x_values[tolerance_indices.max()]

                # Use axvspan to shade the region
                ax.axvspan(x_min,x_max,color='yellow',alpha=0.3,label='Tolerance Area')
                
class SecantMethod(NumericMethod):
    def __init__(self, graph_plotter):
        self.plotter = graph_plotter
        self.function = self.plotter.function
        self.a = self.plotter.a
        self.b = self.plotter.b
        self.x0 = self.a
        self.x1 = self.b
        self.attempt = 0
        self.steps = []
        self.iteration_data = []
        self.create_iteration_window()

    def step(self):
        f_x0 = self.function(self.x0)
        f_x1 = self.function(self.x1)
        
        x_new = self.x1 - f_x1 * (self.x1 - self.x0) / (f_x1 - f_x0)
        
        error = np.abs(x_new - self.x1)
        
        self.attempt += 1
        
        self.iteration_data.append((self.attempt, self.x0, self.x1, self.x1, f_x1, error))
        
        self.x0 = self.x1
        self.x1 = x_new
        
        self.steps.append(self.x1)
        

    def plot_step(self):
        """Plots the current step of the secant method using GraphPlotter."""
        self.plotter.update()

        # Plot the interval and secant line
        self.plotter.ax.axvline(x=self.x0, color='red', linestyle='--', label=f'x0 = {self.x0}')
        self.plotter.ax.axvline(x=self.x1, color='green', linestyle='--', label=f'x1 = {self.x1}')

        # x values rea the visible x range
        x_values = np.linspace(self.plotter.ax.get_xlim()[0], self.plotter.ax.get_xlim()[1], 1000)
        y_values = self.function(x_values)
        
        secant_line = self.function(self.x1) + (self.function(self.x1) - self.function(self.x0)) / (self.x1 - self.x0) * (x_values - self.x1)
        self.plotter.ax.plot(x_values, secant_line, color='purple', linestyle='--', label='Secant Line')

        self.plotter.plot_tolerance_area(self.plotter.ax, x_values, y_values, not_in_interval=True)
        
        self.plotter.fill_areas(self.x0, self.x1)

        self.plotter.ax.legend()
        self.plotter.canvas.draw()

    def create_iteration_window(self):
        """Creates a new window to perform iterations of the secant method."""
        # Create a top-level window with a small initial size
        iteration_window = tk.Toplevel(self.plotter.root)
        iteration_window.title("Secant Method Iteration")
        iteration_window.geometry("500x400")
        iteration_window.configure(bg="#f0f0f0")
        
        # Keep window on top of the main window
        iteration_window.attributes("-topmost", True)
        
        # Title label for the window, centered and styled
        title_label = tk.Label(iteration_window, text="Secant Method Iteration", font=("Helvetica", 14, "bold"), bg="#f0f0f0", anchor="center")
        title_label.pack(fill=tk.X, pady=(10, 5))
        
        # Label to display the current iteration details
        self.iteration_label = tk.Label(iteration_window, text="Current iteration details", font=("Helvetica", 12), bg="#f0f0f0", anchor="center")
        self.iteration_label.pack(fill=tk.X, pady=(5, 10))
        
        # Add a button for the next iteration
        self.next_button = tk.Button(iteration_window, text="Next Iteration", command=self.next_iteration, font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", activebackground="#005f99", activeforeground="white", relief="raised")
        self.next_button.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Frame to hold iteration details
        iteration_frame = tk.Frame(iteration_window, bg="#f0f0f0")
        iteration_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a table-like structure for displaying iteration details
        headers = ["Attempt", "x0", "x1", "x2", "f(x1)", "Error"]
        for col, header in enumerate(headers):
            label = tk.Label(iteration_frame, text=header, font=("Helvetica", 12, "bold"), bg="#d3d3d3", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid")
            label.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            iteration_frame.grid_columnconfigure(col, weight=1, uniform="col")
        
        self.iteration_table = tk.Frame(iteration_frame, bg="#f0f0f0")
        self.iteration_table.grid(row=1, column=0, columnspan=len(headers), sticky="nsew")
        iteration_frame.grid_rowconfigure(1, weight=1)
        
    def update_iteration_table(self):
        """Updates the table with the latest iteration details."""
        for widget in self.iteration_table.winfo_children():
            widget.destroy()
        
        for i, (attempt, x0, x1, x2, f_x1, error) in enumerate(self.iteration_data):
            tk.Label(self.iteration_table, text=str(attempt), font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{x0:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{x1:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{x2:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=3, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{f_x1:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=4, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{error:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=5, sticky="nsew", padx=1, pady=1)
            
        for col in range(6):
            self.iteration_table.grid_columnconfigure(col, weight=1, uniform="col")
        
    def next_iteration(self):
        """Performs the next iteration of the secant method and updates the plot."""
        self.step()
        
        self.plot_step()
        
        self.iteration_label.config(text=f"Attempt: {self.iteration_data[-1][0]} | x2: {self.iteration_data[-1][3]:.4f} | Error: {self.iteration_data[-1][5]:.4f}")
        
        self.update_iteration_table()
    
class TrapezoidalMethod(NumericMethod):
    def __init__(self, graph_plotter):
        self.plotter = graph_plotter
        self.function = self.plotter.function
        self.a = self.plotter.a
        self.b = self.plotter.b
        self.n = 1
        self.h = (self.b - self.a) / self.n
        self.tolerance = self.plotter.tolerance
        self.attempt = 0
        self.iteration_data = []
        self.create_iteration_window()

    def step(self):
        self.h = (self.b - self.a) / self.n
        x0 = self.a
        sum_of_trapezoids = 0
        
        for _ in range(self.n):
            xi = x0 + self.h
            sum_of_trapezoids += (self.function(x0) + self.function(xi)) / 2
            x0 = xi
        
        integral = self.h * sum_of_trapezoids
        
        self.attempt += 1
        
        error = (self.b - self.a) * (self.h ** 2) / 12
        
        self.iteration_data.append((self.attempt, self.n, integral, error))
        
        return integral

    def plot_step(self):
        """Plots the current step of the trapezoidal method using GraphPlotter."""
        self.plotter.update()

        x_values = np.linspace(self.a, self.b, self.n + 1)
        y_values = self.function(x_values)

        # Plot each trapezoid
        for i in range(self.n):
            x_trap = [x_values[i], x_values[i], x_values[i + 1], x_values[i + 1]]
            y_trap = [0, y_values[i], y_values[i + 1], 0]
            self.plotter.ax.fill(x_trap, y_trap, color='blue', edgecolor='black', alpha=0.3)

        self.plotter.ax.text(self.a, 3, 'A', fontsize=12, color='red', ha='center')
        self.plotter.ax.text(self.b, 3, 'B', fontsize=12, color='green', ha='center')

        self.plotter.ax.text(self.a, -1, f'{self.a}', fontsize=12, color='red', ha='center')
        self.plotter.ax.text(self.b, -1, f'{self.b}', fontsize=12, color='green', ha='center')

        self.plotter.ax.axvline(x=self.a, color='red', linestyle='--', label=f'a = {self.a}')
        self.plotter.ax.axvline(x=self.b, color='green', linestyle='--', label=f'b = {self.b}')
        
        self.plotter.ax.legend()
        self.plotter.canvas.draw()
        
    def create_iteration_window(self):
        """Creates a new window to perform iterations of the trapezoidal method."""
        # Create a top-level window with a small initial size
        iteration_window = tk.Toplevel(self.plotter.root)
        iteration_window.title("Trapezoidal Method Iteration")
        iteration_window.geometry("500x600")
        iteration_window.configure(bg="#f0f0f0")
        
        # Keep window on top of the main window
        iteration_window.attributes("-topmost", True)

        # Title label for the window, centered and styled
        title_label = tk.Label(iteration_window, text="Trapezoidal Method Iteration", font=("Helvetica", 14, "bold"), bg="#f0f0f0", anchor="center")
        title_label.pack(fill=tk.X, pady=(10, 5))

        # Label to display the current iteration details
        self.iteration_label = tk.Label(iteration_window, text="Current iteration details", font=("Helvetica", 12), bg="#f0f0f0", anchor="center")
        self.iteration_label.pack(fill=tk.X, pady=(5, 10))

        # Slider to update n
        n_slider_label = tk.Label(iteration_window, text="Update n value: ", font=("Helvetica", 12), bg="#f0f0f0")
        n_slider_label.pack(pady=(5, 5))
        self.n_slider = tk.Scale(iteration_window, from_=1, to=100, orient=tk.HORIZONTAL, length=300)
        self.n_slider.set(self.n)
        self.n_slider.pack(pady=(5, 10))

        # Button to update n and perform iteration
        update_n_button = tk.Button(iteration_window, text="Update n and Recalculate", command=self.next_iteration, font=("Helvetica", 12, "bold"), bg="#007acc", fg="white", activebackground="#005f99", activeforeground="white", relief="raised")
        update_n_button.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Frame to hold iteration details
        iteration_frame = tk.Frame(iteration_window, bg="#f0f0f0")
        iteration_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a table-like structure for displaying iteration details
        headers = ["Attempt", "n", "Integral", "Error"]
        for col, header in enumerate(headers):
            label = tk.Label(iteration_frame, text=header, font=("Helvetica", 12, "bold"), bg="#d3d3d3", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid")
            label.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            iteration_frame.grid_columnconfigure(col, weight=1, uniform="col")  # Set equal weight and uniform group
        
        self.iteration_table = tk.Frame(iteration_frame, bg="#f0f0f0")
        self.iteration_table.grid(row=1, column=0, columnspan=len(headers), sticky="nsew")
        iteration_frame.grid_rowconfigure(1, weight=1)
        
        self.next_iteration()

    def update_iteration_table(self):
        """Updates the table with the latest iteration details."""
        for widget in self.iteration_table.winfo_children():
            widget.destroy()

        for i, (attempt, n, integral, error) in enumerate(self.iteration_data):
            tk.Label(self.iteration_table, text=str(attempt), font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=str(n), font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{integral:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(self.iteration_table, text=f"{error:.4f}", font=("Helvetica", 12), bg="#f0f0f0", anchor="center", padx=5, pady=5, borderwidth=1, relief="solid").grid(row=i, column=3, sticky="nsew", padx=1, pady=1)

        # Configure columns to have equal weight and uniform group
        for col in range(4):
            self.iteration_table.grid_columnconfigure(col, weight=1, uniform="col")
        
    def next_iteration(self):
        """Performs the next iteration of the trapezoidal method with the new n and updates the plot."""
        self.n = int(self.n_slider.get())
        integral = self.step()
        self.plot_step()
        self.iteration_label.config(text=f"Attempt: {self.iteration_data[-1][0]} | n: {self.iteration_data[-1][1]} | Integral: {integral:.4f} | Error: {self.iteration_data[-1][3]:.4f}")
        self.update_iteration_table()
        
class RungeKuttaMethod(NumericMethod):
    def __init__(self, graph_plotter):
        self.plotter = graph_plotter
        self.function = lambda x, y: (x - y) / 2 # Placeholder for user-defined function
        self.function_str = "(x - y) / 2"
        self.tolerance = self.plotter.tolerance
        self.h = .2  # Step size
        self.x0 = 0
        self.y0 = 1
        self.x = 2
        self.attempt = 0
        self.iteration_data_of_each_order = {1: [], 2: [], 3: [], 4: []}
        self.create_iteration_window()

    def step(self):
        """Performs the Runge-Kutta calculation for the given function."""
        n = int((self.x - self.x0) / self.h)
        x = self.x0
        y = self.y0

        # 1st order Runge-Kutta method (Euler's method)
        for i in range(n):
            k1 = self.h * self.function(x, y)
            y = y + k1
            x = x + self.h
            self.iteration_data_of_each_order[1].append((x, y))

        x, y = self.x0, self.y0  # Reset for 2nd-order

        # 2nd order Runge-Kutta method (Midpoint method)
        for i in range(n):
            k1 = self.h * self.function(x, y)
            k2 = self.h * self.function(x + 0.5 * self.h, y + 0.5 * k1)
            y = y + k2
            x = x + self.h
            self.iteration_data_of_each_order[2].append((x, y))

        x, y = self.x0, self.y0  # Reset for 3rd-order

        # 3rd order Runge-Kutta method
        for i in range(n):
            k1 = self.h * self.function(x, y)
            k2 = self.h * self.function(x + 0.5 * self.h, y + 0.5 * k1)
            k3 = self.h * self.function(x + self.h, y - k1 + 2 * k2)  # Adjusted k3
            y = y + (1.0 / 6.0) * (k1 + 4 * k2 + k3)
            x = x + self.h
            self.iteration_data_of_each_order[3].append((x, y))

        x, y = self.x0, self.y0  # Reset for 4th-order

        # 4th order Runge-Kutta method
        for i in range(n):
            k1 = self.h * self.function(x, y)
            k2 = self.h * self.function(x + 0.5 * self.h, y + 0.5 * k1)
            k3 = self.h * self.function(x + 0.5 * self.h, y + 0.5 * k2)
            k4 = self.h * self.function(x + self.h, y + k3)
            y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + self.h
            self.iteration_data_of_each_order[4].append((x, y))

        return y

    def create_iteration_window(self):
        """Creates a GUI window for inputting the necessary parameters for the Runge-Kutta method and displays the results."""
        # Check if the window already exists
        if hasattr(self, 'iteration_window') and self.iteration_window.winfo_exists():
            self.iteration_window.destroy()  # Destroy if already exists to create a fresh instance

        # Create the main window
        self.iteration_window = tk.Toplevel(self.plotter.root)
        self.iteration_window.title("Runge-Kutta Method Parameters")
        self.iteration_window.geometry("500x500")
        self.iteration_window.configure(bg="#f7f7f7")

        # Title label with a bold style
        tk.Label(self.iteration_window, text="Runge-Kutta Method Parameters",
                font=("Helvetica", 16, "bold"), bg="#f7f7f7", fg="#333").pack(pady=(10, 5))

        # Instruction label
        self.instruction_label = tk.Label(self.iteration_window,
                                        text="Enter the following values to calculate the Runge-Kutta method:",
                                        font=("Helvetica", 12), bg="#f7f7f7", fg="#555")
        self.instruction_label.pack(pady=(0, 15))

        # Frame for input fields
        self.input_frame = tk.Frame(self.iteration_window, bg="#f7f7f7")
        self.input_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Input fields
        self.create_input_fields()

    def create_input_fields(self):
        """Creates input fields for Runge-Kutta parameters."""
        # Function input
        tk.Label(self.input_frame, text="Function f(x, y):", font=("Helvetica", 12), bg="#f7f7f7").pack(anchor="w", pady=(5, 2))
        self.function_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.function_entry.insert(0, self.function_str)
        self.function_entry.pack(fill=tk.X, pady=(0, 10))

        # x0 input
        tk.Label(self.input_frame, text="Initial x0:", font=("Helvetica", 12), bg="#f7f7f7").pack(anchor="w", pady=(5, 2))
        self.x0_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.x0_entry.insert(0, self.x0)
        self.x0_entry.pack(fill=tk.X, pady=(0, 10))

        # y0 input
        tk.Label(self.input_frame, text="Initial y0:", font=("Helvetica", 12), bg="#f7f7f7").pack(anchor="w", pady=(5, 2))
        self.y0_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.y0_entry.insert(0, self.y0)
        self.y0_entry.pack(fill=tk.X, pady=(0, 10))

        # Final x input
        tk.Label(self.input_frame, text="Final x value:", font=("Helvetica", 12), bg="#f7f7f7").pack(anchor="w", pady=(5, 2))
        self.x_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.x_entry.insert(0, self.x)
        self.x_entry.pack(fill=tk.X, pady=(0, 10))

        # Step size h input
        tk.Label(self.input_frame, text="Step size (h):", font=("Helvetica", 12), bg="#f7f7f7").pack(anchor="w", pady=(5, 2))
        self.h_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.h_entry.insert(0, self.h)
        self.h_entry.pack(fill=tk.X, pady=(0, 15))

        # Submit button
        submit_button = tk.Button(self.input_frame, text="Run Method", command=self.next_iteration,
                                font=("Helvetica", 12, "bold"), bg="#007acc", fg="white",
                                activebackground="#005f99", activeforeground="white", relief="raised")
        submit_button.pack(fill=tk.X, pady=(10, 20))

    def next_iteration(self):
        """Executes the Runge-Kutta method and displays the result."""
        try:
            # Parse inputs
            self.func_str = self.function_entry.get()
            self.function = eval(f"lambda x, y: {self.func_str}")
            self.x0 = float(self.x0_entry.get())
            self.y0 = float(self.y0_entry.get())
            self.x = float(self.x_entry.get())
            self.h = float(self.h_entry.get())

            # Perform the calculation
            result = self.step()

            # Clear input fields and show the result
            self.input_frame.destroy()
            self.instruction_label.config(text="Runge-Kutta Calculation Result:")
            
            self.plot_step()
            
            # Display the result
            result_label = tk.Label(self.iteration_window, text=f"Result: y({self.x}) = {result}",
                                    font=("Helvetica", 14, "bold"), bg="#f7f7f7", fg="#333")
            result_label.pack(pady=(20, 10))

        except Exception as e:
            messagebox.showerror("Input Error", f"Error in inputs: {e}")

                
    def plot_step(self):
        """
        Plots the results of the Runge-Kutta methods for each order using the GraphPlotter instance.
        """
        orders = [1, 2, 3, 4]  # Runge-Kutta orders
        titles = {
            1: "1st Order (Euler's Method)",
            2: "2nd Order Runge-Kutta",
            3: "3rd Order Runge-Kutta",
            4: "4th Order Runge-Kutta",
        }
        
        # Clear the plot
        self.plotter.clear_plot()
        self.plotter.setup_grid()
        
        # Cler function plot not to overlap with the Runge-Kutta method
        self.plotter.plot[0].remove()

        for idx, order in enumerate(orders):
            if order in self.iteration_data_of_each_order:
                x_values, y_values = zip(*self.iteration_data_of_each_order[order])  # Unpack data
                self.plotter.ax.plot(x_values, y_values, marker='o', label=f'{order} Order RK', color=f'C{order}')
                self.plotter.ax.set_title(titles[order])
                self.plotter.ax.set_xlabel('x')
                self.plotter.ax.set_ylabel('y')
                self.plotter.ax.legend()
                self.plotter.ax.grid(True)
            else:
                self.plotter.ax.text(0.5, 0.5, 'No Data', transform=self.ax.transAxes, ha='center', va='center')

        # Redraw the updated canvas in the GUI
        self.plotter.canvas.draw()
          
class GraphicalInterface:
    def __init__(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Interactive Graph with Controls")
        self.root.geometry("1200x900")
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

        # Create the main frame to hold all content
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)

        # Create the plotter frame (for graph plotting)
        self.plotter_frame = tk.Frame(self.main_frame)
        self.plotter_frame.grid(row=0, column=1, sticky="nsew")
        self.plotter_frame.grid_rowconfigure(0, weight=1)
        self.plotter_frame.grid_columnconfigure(0, weight=1)

        # Initialize the function to plot
        self.function = lambda x: 1 / x - 1
        self.plotter = GraphPlotter(self.plotter_frame, function=self.function)

        # Create the control frame for user input
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # Create and set up settings frame for changing function and intervals
        self._create_settings_frame()

        # Create and set up methods frame for numerical methods
        self._create_methods_frame()

        # Run the main loop
        self.root.mainloop()

    def _create_settings_frame(self):
        """Creates the settings frame for changing the function and interval values."""
        self.settings_frame = tk.LabelFrame(self.control_frame, text="Settings", padx=10, pady=10)
        self.settings_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        # Reload button for resetting the graph
        self.reload_image = Image.open("./Reload.png").resize((20, 20))
        self.reload_photo = ImageTk.PhotoImage(self.reload_image)
        self.reload_button = tk.Button(self.plotter_frame, text="", command=self.plotter.plot_function, image=self.reload_photo, compound="left")
        self.reload_button.place(relx=1.0, rely=0.0, anchor="ne", x=-50, y=50)

        # Settings title
        tk.Label(self.settings_frame, text="Change Settings", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Function input field
        tk.Label(self.settings_frame, text="Enter function f(x):").grid(row=1, column=0, pady=5, sticky="w")
        self.function_entry = tk.Entry(self.settings_frame)
        self.function_entry.grid(row=2, column=0, pady=5, sticky="ew", columnspan=2)
        self.function_entry.insert(0, self.plotter.function_str)

        # Interval 'a' input
        tk.Label(self.settings_frame, text="Enter interval value for a:").grid(row=3, column=0, pady=5, sticky="w")
        self.entry_a = tk.Entry(self.settings_frame, width=15)
        self.entry_a.grid(row=4, column=0, pady=5, sticky="ew", columnspan=2)
        self.entry_a.insert(0, str(self.plotter.a))

        # Interval 'b' input
        tk.Label(self.settings_frame, text="Enter interval value for b:").grid(row=5, column=0, pady=5, sticky="w")
        self.entry_b = tk.Entry(self.settings_frame, width=15)
        self.entry_b.grid(row=6, column=0, pady=5, sticky="ew", columnspan=2)
        self.entry_b.insert(0, str(self.plotter.b))

        tk.Label(self.settings_frame, text="Enter tolerance value:").grid(row=7, column=0, pady=5, sticky="w")
        self.entry_tolerance = tk.Entry(self.settings_frame, width=15)
        self.entry_tolerance.grid(row=8, column=0, pady=5, sticky="ew", columnspan=2)
        self.entry_tolerance.insert(0, "0.01")

        # Submit button to update the graph with new function or intervals
        submit_button = tk.Button(self.settings_frame, text="Update Graph", command=self.update_graph)
        submit_button.grid(row=9, column=0, pady=10, sticky="ew", columnspan=2)

        reset_button = tk.Button(self.control_frame, text="Reset Graph", command=self.plotter.reset_graph)
        reset_button.grid(row=2, column=0, pady=10, sticky="ew", padx=10)

    def _create_methods_frame(self):
        """Creates the methods frame for selecting numerical methods."""
        self.methods_frame = tk.LabelFrame(self.control_frame, text="Methods", padx=10, pady=10)
        self.methods_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # Methods title
        tk.Label(self.methods_frame, text="Choose Method", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # List of available methods with their respective properties
        methods = [
            ("Bisection", lambda: BisectionMethod(self.plotter), 1, 0, (0, 5)),
            ("Newton", lambda: NewtonMethod(self.plotter), 1, 1, (5, 0)),
            ("Secante", lambda: SecantMethod(self.plotter), 2, 0, (0, 5)),
            ("Punto Fijo", lambda: NewtonMethod(self.plotter), 2, 1, (5, 0)),
            ("Lagrange", lambda: BisectionMethod(self.plotter), 3, 0, (0, 5)),
            ("Regresion", lambda: NewtonMethod(self.plotter), 3, 1, (5, 0)),
            ("Diferencias Divididas", lambda: BisectionMethod(self.plotter), 4, 0, (0, 5)),
            ("Derivacion Numerica", lambda: NewtonMethod(self.plotter), 4, 1, (5, 0)),
            ("Trapecio", lambda: TrapezoidalMethod(self.plotter), 5, 0, (0, 5)),
            ("Simpson", lambda: BisectionMethod(self.plotter), 5, 1, (5, 0)),
            ("Romberg", lambda: NewtonMethod(self.plotter), 6, 0, (0, 5)),
            ("Runge Kutta", lambda: RungeKuttaMethod(self.plotter), 6, 1, (5, 0)),
            ("Euler", lambda: NewtonMethod(self.plotter), 7, 0, (0, 5)),
        ]

        # Create buttons for each method
        for text, command, row, column, padx in methods:
            button = tk.Button(self.methods_frame, text=text, command=command)
            button.grid(row=row, column=column, pady=10, padx=padx, sticky="ew")

        # Configure columns for equal weight distribution
        self.methods_frame.grid_columnconfigure(0, weight=1)
        self.methods_frame.grid_columnconfigure(1, weight=1)

    def update_graph(self):
        """Updates the function or interval values and re-plots the graph."""
        try:
            # Update the function string and lambda
            function_str = self.function_entry.get()
            self.plotter.function_str = function_str
            
            # Parse the function string using sympy
            x = sy.symbols('x')
            expr = sy.sympify(function_str)
            
            # Convert the parsed expression to a lambda function
            self.plotter.function = sy.lambdify(x, expr, 'numpy')

            # Update interval values
            self.plotter.a = float(self.entry_a.get())
            self.plotter.b = float(self.entry_b.get())
            self.plotter.tolerance = float(self.entry_tolerance.get())

            # Validate the interval values
            if self.plotter.a >= self.plotter.b:
                messagebox.showerror("Input Error", "Value of 'a' must be less than 'b'")
                return

            # Update the graph plotter with new function
            self.plotter.update()
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            
# To run the graphical interface
if __name__ == "__main__":
    GraphicalInterface()

