# Author: Lily Williams
# October 2024
# Heavily modified from code written by EdX User soundsoffailure

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.sparse import spdiags, csr_matrix
import matplotlib.animation as animation
from numba import jit, njit
from typing import Tuple
import threading



class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Two-Stream Plasma Instability Simulation")
        
        # Initialize theme state
        self.is_dark_mode = tk.BooleanVar(value=False)
        
        # Create main containers
        self.left_frame = ttk.Frame(root, padding="10")
        self.right_frame = ttk.Frame(root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure grid weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        
        # Initialize visualization parameters
        self.viz_params = {
            'marker_size': tk.DoubleVar(value=1.0),
            'stream1_color': tk.StringVar(value="firebrick"),
            'stream2_color': tk.StringVar(value="dodgerblue"),
            'alpha': tk.DoubleVar(value=1.0),
            'phase_xlim': tk.DoubleVar(value=3.141),
            'phase_ylim': tk.DoubleVar(value=2.0),
        }
        
        # Create and configure styles
        self._create_styles()
        
        self._create_parameter_inputs()
        self._create_viz_controls()
        self._create_plot_area()
        self._create_control_buttons()
        
        # Initialize simulation state
        self.simulation = None
        self.animation = None
        self.is_running = False
        
        # Add theme toggle button
        self._create_theme_toggle()
        
        # Apply initial theme
        self._apply_theme()

    def _create_parameter_inputs(self):
        """Create input fields for simulation parameters"""
        # Create a labeled frame for simulation parameters
        sim_frame = ttk.LabelFrame(self.left_frame, text="Simulation Parameters", padding="5")
        sim_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Parameters with their default values and ranges
        self.parameters = {
            'L': {'label': 'System Length (L):', 'default': np.pi, 'from_': 1, 'to': 10},
            'DT': {'label': 'Time Step (DT):', 'default': 0.25, 'from_': 0.01, 'to': 1.0},
            'NT': {'label': 'Number of Time Steps:', 'default': 400, 'from_': 100, 'to': 1000},
            'NG': {'label': 'Grid Points:', 'default': 32, 'from_': 16, 'to': 128},
            'N': {'label': 'Number of Particles:', 'default': 1000, 'from_': 100, 'to': 5000},
            'V0': {'label': 'Initial Velocity:', 'default': 0.2, 'from_': 0.1, 'to': 1.0},
            'VT': {'label': 'Thermal Velocity:', 'default': 0.0, 'from_': 0.0, 'to': 0.5},
            'XP1': {'label': 'Position Perturbation:', 'default': 1, 'from_': 0, 'to': 2}
        }
        
        # Create and arrange parameter inputs
        self.param_vars = {}
        for i, (param, info) in enumerate(self.parameters.items()):
            ttk.Label(sim_frame, text=info['label']).grid(row=i, column=0, padx=5, pady=2, sticky='e')
            var = tk.DoubleVar(value=info['default'])
            self.param_vars[param] = var
            
            spinbox = ttk.Spinbox(
                sim_frame,
                from_=info['from_'],
                to=info['to'],
                textvariable=var,
                width=10,
                increment=0.1 if param in ['DT', 'V0', 'VT', 'XP1'] else 1
            )
            spinbox.grid(row=i, column=1, padx=5, pady=2, sticky='w')

    def _create_viz_controls(self):
        """Create controls for visualization parameters"""
        # Create a labeled frame for visualization parameters
        viz_frame = ttk.LabelFrame(self.left_frame, text="Visualization Controls", padding="5")
        viz_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Marker size control
        ttk.Label(viz_frame, text="Marker Size:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        marker_size = ttk.Spinbox(
            viz_frame,
            from_=0.1,
            to=10.0,
            textvariable=self.viz_params['marker_size'],
            width=10,
            increment=0.1
        )
        marker_size.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        
        # Alpha (transparency) control
        ttk.Label(viz_frame, text="Opacity:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        alpha = ttk.Spinbox(
            viz_frame,
            from_=0.1,
            to=1.0,
            textvariable=self.viz_params['alpha'],
            width=10,
            increment=0.1
        )
        alpha.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        
        # Color picker buttons
        ttk.Button(
            viz_frame,
            text="Stream 1 Color",
            command=lambda: self._pick_color('stream1_color')
        ).grid(row=2, column=0, columnspan=2, pady=2)
        
        ttk.Button(
            viz_frame,
            text="Stream 2 Color",
            command=lambda: self._pick_color('stream2_color')
        ).grid(row=3, column=0, columnspan=2, pady=2)
        
        # Plot limits
        ttk.Label(viz_frame, text="Phase Space X Limit:").grid(row=4, column=0, padx=5, pady=2, sticky='e')
        x_limit = ttk.Spinbox(
            viz_frame,
            from_=1.0,
            to=20.0,
            textvariable=self.viz_params['phase_xlim'],
            width=10,
            increment=0.5
        )
        x_limit.grid(row=4, column=1, padx=5, pady=2, sticky='w')
        
        ttk.Label(viz_frame, text="Phase Space Y Limit:").grid(row=5, column=0, padx=5, pady=2, sticky='e')
        y_limit = ttk.Spinbox(
            viz_frame,
            from_=0.5,
            to=10.0,
            textvariable=self.viz_params['phase_ylim'],
            width=10,
            increment=0.5
        )
        y_limit.grid(row=5, column=1, padx=5, pady=2, sticky='w')

    def _pick_color(self, color_var):
        """Open color picker and update color variable"""
        color = colorchooser.askcolor(color=self.viz_params[color_var].get())
        if color[1]:  # If a color was picked (not cancelled)
            self.viz_params[color_var].set(color[1])

    def _create_plot_area(self):
        """Create matplotlib figures for phase space and energy plots"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Phase space tab
        self.phase_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phase_frame, text="Phase Space")
        
        # Energy plot tab
        self.energy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.energy_frame, text="Energy")
        
        # Create figures with appropriate colors
        colors = self.dark_colors if self.is_dark_mode.get() else self.light_colors
        
        self.phase_fig = Figure(figsize=(8, 6), facecolor=colors['bg'])
        self.energy_fig = Figure(figsize=(8, 6), facecolor=colors['bg'])
        
        # Phase space plot
        self.phase_ax = self.phase_fig.add_subplot(111)
        self.phase_ax.set_facecolor(colors['frame_bg'])
        self.phase_canvas = FigureCanvasTkAgg(self.phase_fig, self.phase_frame)
        self.phase_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Energy plot
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_ax.set_facecolor(colors['frame_bg'])
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, self.energy_frame)
        self.energy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Apply initial theme
        self._apply_theme()

    def _create_control_buttons(self):
        """Create control buttons for the simulation"""
        button_frame = ttk.Frame(self.left_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        self.run_button = ttk.Button(
            button_frame,
            text="Run Simulation",
            command=self.run_simulation,
            style='Run.TButton'
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_simulation,
            style='Stop.TButton',
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def run_simulation(self):
        """Start the simulation with current parameters"""
        if self.is_running:
            return
            
        # Get parameters from input fields
        try:
            config = {
                param: int(var.get()) if param in ['NT', 'NG', 'N'] else var.get() 
                for param, var in self.param_vars.items()
            }
            config['dens'] = 1  # Fixed parameter
            
            # Add visualization parameters
            config['viz'] = {
                'marker_size': self.viz_params['marker_size'].get(),
                'stream1_color': self.viz_params['stream1_color'].get(),
                'stream2_color': self.viz_params['stream2_color'].get(),
                'alpha': self.viz_params['alpha'].get(),
                'phase_xlim': self.viz_params['phase_xlim'].get(),
                'phase_ylim': self.viz_params['phase_ylim'].get(),
            }
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")
            return
            
        # Disable run button and enable stop button
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.is_running = True
        
        # Clear previous plots
        self.phase_ax.clear()
        self.energy_ax.clear()
        
        # Create and run simulation in a separate thread
        self.simulation = PlasmaSimulation(config, gui_mode=True)
        threading.Thread(target=self._run_simulation_thread, daemon=True).start()

    def _run_simulation_thread(self):
        """Run simulation in a separate thread to prevent GUI freezing"""
        try:
            self.simulation.run_simulation(
                phase_ax=self.phase_ax,
                energy_ax=self.energy_ax,
                phase_canvas=self.phase_canvas,
                energy_canvas=self.energy_canvas
            )
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation error: {str(e)}"))
        finally:
            self.root.after(0, self._simulation_completed)

    def _simulation_completed(self):
        """Reset GUI state after simulation completes"""
        self.is_running = False
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation:
            self.simulation.stop_simulation()
        self._simulation_completed()

    def _create_styles(self):
        """Create and configure ttk styles for both light and dark themes"""
        self.style = ttk.Style()
        
        # Light theme colors
        self.light_colors = {
            'bg': '#ffffff',
            'fg': '#000000',
            'frame_bg': '#f0f0f0',
            'button_bg': '#e0e0e0',
            'accent': '#0078d7'
        }
        
        # Dark theme colors
        self.dark_colors = {
            'bg': '#2d2d2d',
            'fg': '#ffffff',
            'frame_bg': '#363636',
            'button_bg': '#404040',
            'accent': '#0078d7'
        }
        
        # Configure styles for both themes
        for theme in ['light', 'dark']:
            colors = self.light_colors if theme == 'light' else self.dark_colors
            
            self.style.configure(f'{theme}.TFrame', background=colors['bg'])
            self.style.configure(f'{theme}.TLabelframe', background=colors['bg'])
            self.style.configure(f'{theme}.TLabelframe.Label', 
                               background=colors['bg'],
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.TLabel', 
                               background=colors['bg'],
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.TButton', 
                               background=colors['button_bg'],
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.Run.TButton',
                               background='green',
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.Stop.TButton',
                               background='red',
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.TSpinbox',
                               fieldbackground=colors['button_bg'],
                               foreground=colors['fg'])
            self.style.configure(f'{theme}.TNotebook',
                               background=colors['bg'])
            self.style.configure(f'{theme}.TNotebook.Tab',
                               background=colors['button_bg'],
                               foreground=colors['fg'])

    def _apply_theme(self):
        """Apply the current theme to all widgets"""
        theme = 'dark' if self.is_dark_mode.get() else 'light'
        colors = self.dark_colors if self.is_dark_mode.get() else self.light_colors
        
        # Update root window
        self.root.configure(bg=colors['bg'])
        
        # Update all frames
        for frame in [self.left_frame, self.right_frame]:
            frame.configure(style=f'{theme}.TFrame')
        
        # Update all widgets
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                widget.configure(style=f'{theme}.TLabelframe')
            elif isinstance(widget, ttk.Label):
                widget.configure(style=f'{theme}.TLabel')
            elif isinstance(widget, ttk.Button):
                if 'run' in str(widget).lower():
                    widget.configure(style=f'{theme}.Run.TButton')
                elif 'stop' in str(widget).lower():
                    widget.configure(style=f'{theme}.Stop.TButton')
                else:
                    widget.configure(style=f'{theme}.TButton')
            elif isinstance(widget, ttk.Spinbox):
                widget.configure(style=f'{theme}.TSpinbox')
            elif isinstance(widget, ttk.Notebook):
                widget.configure(style=f'{theme}.TNotebook')
        
        # Update plot styles
        if hasattr(self, 'phase_fig'):
            self.phase_fig.set_facecolor(colors['bg'])
            self.phase_ax.set_facecolor(colors['frame_bg'])
            self.phase_ax.tick_params(colors=colors['fg'])
            for spine in self.phase_ax.spines.values():
                spine.set_color(colors['fg'])
            self.phase_ax.xaxis.label.set_color(colors['fg'])
            self.phase_ax.yaxis.label.set_color(colors['fg'])
            self.phase_ax.title.set_color(colors['fg'])
            
        if hasattr(self, 'energy_fig'):
            self.energy_fig.set_facecolor(colors['bg'])
            self.energy_ax.set_facecolor(colors['frame_bg'])
            self.energy_ax.tick_params(colors=colors['fg'])
            for spine in self.energy_ax.spines.values():
                spine.set_color(colors['fg'])
            self.energy_ax.xaxis.label.set_color(colors['fg'])
            self.energy_ax.yaxis.label.set_color(colors['fg'])
            self.energy_ax.title.set_color(colors['fg'])
            if self.energy_ax.get_legend():
                self.energy_ax.get_legend().set_frame_on(True)
                self.energy_ax.get_legend().get_frame().set_facecolor(colors['frame_bg'])
                self.energy_ax.get_legend().get_frame().set_edgecolor(colors['fg'])
                for text in self.energy_ax.get_legend().get_texts():
                    text.set_color(colors['fg'])
        
        # Redraw canvases if they exist
        if hasattr(self, 'phase_canvas'):
            self.phase_canvas.draw()
        if hasattr(self, 'energy_canvas'):
            self.energy_canvas.draw()

    def _create_theme_toggle(self):
        """Create the theme toggle button"""
        self.theme_button = ttk.Button(
            self.left_frame,
            text="Toggle Dark Mode",
            command=self._toggle_theme
        )
        self.theme_button.grid(row=3, column=0, pady=10)

    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        self.is_dark_mode.set(not self.is_dark_mode.get())
        self._apply_theme()


class PlasmaSimulation:
    """Plasma simulation class adapted for GUI operation"""
    def __init__(self, config, gui_mode=False):
        self.config = config
        self.gui_mode = gui_mode
        self.stop_flag = False
        self.viz_params = config.get('viz', {})  # Get visualization parameters
        
        # Ensure integer values for grid-related parameters
        self.config['NG'] = int(self.config['NG'])
        self.config['NT'] = int(self.config['NT'])
        self.config['N'] = int(self.config['N'])
        
        self._initialize_parameters()
        self._initialize_state()
        self._initialize_history()

    def _initialize_parameters(self):
        """Initialize derived simulation parameters and constants."""
        self.alpha_p = self.config['dens'] * self.config['L'] / self.config['N']
        self.rho_back = self.config['dens']
        self.dx = self.config['L'] / self.config['NG']
        self._setup_poisson_matrix()

    def _setup_poisson_matrix(self):
        """Construct the finite difference matrix for solving Poisson's equation."""
        ng = int(self.config['NG'])
        un = np.ones(ng - 1)
        diagonals = [un, -2 * un, un]
        self.Poisson = spdiags(diagonals, [-1, 0, 1], ng - 1, ng - 1).toarray()

    def _initialize_state(self):
        """Initialize particle positions and velocities."""
        self.xp = np.linspace(0, self.config['L'] - self.config['L'] / self.config['N'], 
                             self.config['N']).reshape(-1, 1)
        self.vp = self.config['VT'] * np.random.randn(self.config['N'], 1)
        pm = 1 - 2 * (np.arange(1, self.config['N'] + 1).reshape(-1, 1) % 2)
        self.vp = self.vp + pm * self.config['V0']
        self.xp = self.xp + self.config['XP1'] * (self.config['L'] / self.config['N']) * pm * np.random.rand(self.config['N'], 1)
        self.Eg = np.zeros(self.config['NG'])
        self.Phi = 0
        # Use visualization colors from parameters
        self.colors = np.where(
            self.vp.flatten() < 0,
            self.viz_params.get('stream1_color', 'firebrick'),
            self.viz_params.get('stream2_color', 'dodgerblue')
        )

    def _initialize_history(self):
        """Initialize arrays to store system properties."""
        self.mom = np.zeros(self.config['NT'])
        self.E_kin = np.zeros(self.config['NT'])
        self.E_pot = np.zeros(self.config['NT'])
        self.vp_history = np.zeros((len(self.vp), self.config['NT']))
        self.Eg_history = np.zeros((self.config['NG'], self.config['NT']))

    def _compute_field(self):
        """Compute electric field from particle positions."""
        g1 = np.floor(self.xp / self.dx).astype(int)
        g = np.vstack((g1, g1 + 1))
        fraz1 = 1 - np.abs(self.xp / self.dx - g1)
        fraz = np.vstack((fraz1, 1 - fraz1))
        g[g < 0] += self.config['NG']
        g[g >= self.config['NG']] -= self.config['NG']
        p = np.arange(self.config['N'])
        self.mat = csr_matrix((fraz.flatten(), 
                             (np.repeat(p, 2), g.flatten())), 
                             shape=(self.config['N'], self.config['NG']))
        rho = -self.alpha_p / self.dx * np.sum(self.mat.toarray(), axis=0) + self.rho_back
        self.Phi = np.linalg.solve(self.Poisson, -rho[: self.config['NG'] - 1] * self.dx**2)
        self.Phi = np.append(self.Phi, 0)
        self.Eg = (np.roll(self.Phi, -1) - np.roll(self.Phi, 1)) / (2 * self.dx)
        self.Eg = self.Eg[:self.config['NG']]

    def update(self, frame):
        """Advance simulation by one timestep and update visualization."""
        self.mom[frame] = np.sum(self.vp) * self.alpha_p
        self.E_kin[frame] = 0.5 * np.sum(self.vp**2) * self.alpha_p
        self.E_pot[frame] = 0.5 * np.sum(self.Eg**2) * self.dx
        self.xp = np.mod(self.xp + self.vp * self.config['DT'], self.config['L'])
        self.vp_history[:, frame] = self.vp.flatten()
        self.Eg_history[:, frame] = np.pad(self.Eg, 
                                         (0, self.config['NG'] - len(self.Eg)), 
                                         "constant")
        self._compute_field()
        self.vp -= (self.mat.dot(self.Eg)).reshape(-1, 1) * self.config['DT']
        if hasattr(self, 'scat'):
            self.scat.set_offsets(np.c_[self.xp, self.vp])
            self.scat.set_color(c=self.colors)
            self.scat.set_sizes(np.full(self.config['N'], 50))
            return (self.scat,)
        return None

    def init_animation(self):
        """Initialize animation with empty plot."""
        self.scat.set_offsets(np.empty((0, 2)))
        return (self.scat,)

    def run_simulation(self, phase_ax=None, energy_ax=None, 
                      phase_canvas=None, energy_canvas=None):
        """Run simulation with real-time plotting updates."""
        for frame in range(self.config['NT']):
            if self.stop_flag:
                break
                
            self.update(frame)
            
            if frame % 5 == 0 and self.gui_mode:
                # Clear and update phase space plot
                phase_ax.clear()
                phase_ax.scatter(
                    self.xp, self.vp, 
                    c=self.colors,
                    s=self.viz_params.get('marker_size', 1.0) * 50,  # Scale marker size
                    alpha=self.viz_params.get('alpha', 1.0)  # Set transparency
                )
                phase_ax.set_xlabel("Position")
                phase_ax.set_ylabel("Velocity")
                phase_ax.set_title(f"Phase Space (t = {frame * self.config['DT']:.2f})")
                
                # Set phase space plot limits
                phase_ax.set_xlim(0, self.viz_params.get('phase_xlim', 10.0))
                phase_ax.set_ylim(-self.viz_params.get('phase_ylim', 2.0),
                                self.viz_params.get('phase_ylim', 2.0))
                
                # Update energy plot
                energy_ax.clear()
                time = np.arange(frame + 1) * self.config['DT']
                energy_ax.plot(time, self.E_kin[:frame + 1], 
                             label="Kinetic", alpha=self.viz_params.get('alpha', 1.0))
                energy_ax.plot(time, self.E_pot[:frame + 1], 
                             label="Potential", alpha=self.viz_params.get('alpha', 1.0))
                energy_ax.plot(time, self.E_kin[:frame + 1] + self.E_pot[:frame + 1], 
                             label="Total", alpha=self.viz_params.get('alpha', 1.0))
                energy_ax.set_xlabel("Time")
                energy_ax.set_ylabel("Energy")
                energy_ax.legend()
                energy_ax.set_title("Energy Evolution")
                
                # Refresh canvases
                phase_canvas.draw()
                energy_canvas.draw()

    def stop_simulation(self):
        """Stop the simulation"""
        self.stop_flag = True


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
