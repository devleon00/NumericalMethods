## Numerical Methods Graphical Interface

This repository contains a Python application built with Tkinter and Matplotlib, showcasing various numerical methods (Bisection, Newton, Secant, Trapezoidal, Runge-Kutta, etc.) and how they converge to solutions (roots of equations, integrals, etc.) in a visual, interactive manner.

---

## Features

- **Interactive Graph**: Pan, zoom, and scroll within the Matplotlib figure embedded in a Tkinter GUI.  
- **Customizable Function**: Change the function `f(x)` directly from the GUI.  
- **Flexible Intervals**: Update the interval `[a, b]` on which to analyze or integrate the function.  
- **Tolerance Setting**: Specify a tolerance value for root-finding methods.  
- **Numerical Methods**:
  - Bisection  
  - Newton-Raphson  
  - Secant  
  - Trapezoidal  
  - Runge-Kutta (1st through 4th order)  
- **Detailed Iteration Windows**: Each method opens a separate iteration window or input window to guide you step-by-step through the calculations, displaying intermediate results in a table-like interface.

---

## Prerequisites

Make sure you have **Python 3.x** installed. Then install the required packages:
```bash
pip install numpy matplotlib sympy Pillow mpl_interactions
```

---

## How to Run

Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Run the main script:

```bash
python main.py
```

Note: If your file has a different name, replace `main.py` accordingly.

A Tkinter window will appear with an interactive Matplotlib canvas:

- **Top panel**: The function visualization.
- **Left panel**: Settings for updating the function, interval `[a, b]`, and tolerance.
- **Methods section**: Buttons to open the iteration windows for each numerical method.

### Zoom/Pan:

- Scroll up/down to zoom in/out where the mouse is pointed.
- Click the pan button on the toolbar to move around the plot.

### Updating the Graph:

- Enter a new function (e.g., `x**2 - 2`) in the “Settings” pane.
- Enter new interval values for `a` and `b`.
- Click `Update Graph` to redraw the function.

### Running a Method:

- Click on one of the method buttons (e.g., Bisection).
- A new window opens, guiding you through iteration steps.
- Intermediate tables and highlighted areas in the graph show the progress of the method.

---

## File Structure

- `main.py` (or the equivalent entry point): Contains the `GraphicalInterface` class, which starts the GUI.
- `GraphPlotter`: Handles plotting, zoom, pan, and redraw features.
- `NumericMethod`: Abstract base class for implementing each numerical method’s iteration logic and plotting steps.
- `BisectionMethod`, `NewtonMethod`, `SecantMethod`, etc.:
  - Step-by-step iteration logic
  - Method-specific plotting
  - Additional GUI windows for input/iteration results

---

## Contributing

Feel free to submit pull requests or open issues. All contributions—ranging from adding features, fixing bugs, to improving documentation—are welcome!