import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import os
import pandas as pd
import re

class BoundaryTracer:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Boundary Tracer - Food Mask with CSV Processing")
        self.root.geometry("1400x900")
        
        # Variables
        self.image = None
        self.photo = None
        self.canvas_image = None
        self.drawing = False
        self.points = []
        self.scale_factor = 1.0
        self.canvas_width = 800
        self.canvas_height = 600
        
        # CSV processing variables
        self.csv_data = None
        self.csv_file_path = None
        self.mask_array = None
        self.original_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image controls (Row 0)
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Clear Trace", command=self.clear_trace).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Create Mask", command=self.create_mask).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Save Mask", command=self.save_mask).grid(row=0, column=3, padx=5)
        
        # CSV controls (Row 1)
        ttk.Button(control_frame, text="Load CSV", command=self.load_csv).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="Load Mask", command=self.load_mask).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Button(control_frame, text="Process CSV", command=self.process_csv).grid(row=1, column=2, padx=5, pady=(5, 0))
        ttk.Button(control_frame, text="Save Results", command=self.save_results).grid(row=1, column=3, padx=5, pady=(5, 0))
        
        # Food interpretation toggle (Row 1)
        self.white_is_food = tk.BooleanVar(value=False)
        food_toggle = ttk.Checkbutton(control_frame, text="White area = ON food", 
                                    variable=self.white_is_food, 
                                    command=self.update_food_interpretation)
        food_toggle.grid(row=1, column=4, padx=5, pady=(5, 0))
        
        # Instructions
        instructions = ttk.Label(control_frame, 
                               text="Instructions: Load image → Trace boundary OR Load existing mask → Create/Load mask → Load CSV → Process CSV → Save results")
        instructions.grid(row=2, column=0, columnspan=5, pady=(10, 0))
        
        # Main content area with two panels
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=2)  # Canvas gets more space
        content_frame.columnconfigure(1, weight=1)  # Info panel gets less space
        content_frame.rowconfigure(0, weight=1)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(content_frame, text="Image Canvas", padding="5")
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="white", width=self.canvas_width, height=self.canvas_height)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Info panel
        info_frame = ttk.LabelFrame(content_frame, text="CSV Information", padding="10")
        info_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(1, weight=1)
        
        # CSV info text
        self.csv_info = tk.Text(info_frame, wrap=tk.WORD, height=15, width=40)
        csv_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.csv_info.yview)
        self.csv_info.configure(yscrollcommand=csv_scrollbar.set)
        
        self.csv_info.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        csv_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Processing results
        self.results_info = tk.Text(info_frame, wrap=tk.WORD, height=10, width=40)
        results_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.results_info.yview)
        self.results_info.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_info.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S), pady=(10, 0))
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image to start")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize info displays
        self.csv_info.insert(tk.END, "No CSV loaded yet.\n\nLoad a CSV file with tracking data to see information here.")
        self.csv_info.config(state=tk.DISABLED)
        
        self.results_info.insert(tk.END, "Processing results will appear here after running 'Process CSV'.")
        self.results_info.config(state=tk.DISABLED)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            try:
                # Load image
                self.image = Image.open(file_path)
                self.original_image = self.image.copy()
                
                # Calculate scale factor to fit canvas while maintaining aspect ratio
                img_width, img_height = self.image.size
                scale_w = self.canvas_width / img_width
                scale_h = self.canvas_height / img_height
                self.scale_factor = min(scale_w, scale_h, 1.0)  # Don't scale up
                
                # Resize image for display
                new_width = int(img_width * self.scale_factor)
                new_height = int(img_height * self.scale_factor)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(self.image)
                
                # Clear canvas and display image
                self.canvas.delete("all")
                self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                
                # Update canvas scroll region
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
                # Clear previous points
                self.points = []
                self.mask_array = None
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)} ({img_width}x{img_height})")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_mask(self):
        """Load an existing mask image"""
        file_path = filedialog.askopenfilename(
            title="Select Mask Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            try:
                if self.original_image is None:
                    messagebox.showwarning("Warning", "Please load the original image first!")
                    return
                
                # Load mask image
                mask_img = Image.open(file_path).convert('L')  # Convert to grayscale
                
                # Resize mask to match original image if needed
                if mask_img.size != self.original_image.size:
                    mask_img = mask_img.resize(self.original_image.size, Image.Resampling.NEAREST)
                    
                # Convert to numpy array
                self.mask_array = np.array(mask_img)
                
                # Clear any existing trace
                self.canvas.delete("trace")
                self.points = []
                
                # Show preview of mask on canvas
                self.show_mask_preview()
                
                self.status_var.set(f"Mask loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load mask: {str(e)}")
    
    def show_mask_preview(self):
        """Show the mask preview on the canvas"""
        if self.mask_array is None or self.original_image is None:
            return
            
        try:
            # Create mask preview for display
            mask_img = Image.fromarray(self.mask_array, mode='L')
            mask_preview = mask_img.resize((int(self.original_image.size[0] * self.scale_factor),
                                          int(self.original_image.size[1] * self.scale_factor)), 
                                         Image.Resampling.NEAREST)
            
            # Create a colored overlay (semi-transparent)
            food_color = (0, 255, 0, 100) if self.white_is_food.get() else (255, 0, 0, 100)  # Green for food, red for no-food
            overlay = Image.new('RGBA', mask_preview.size, (0, 0, 0, 0))  # Transparent background
            
            # Apply overlay only where mask is white
            for y in range(mask_preview.height):
                for x in range(mask_preview.width):
                    if mask_preview.getpixel((x, y)) > 128:  # White area
                        overlay.putpixel((x, y), food_color)
            
            # Composite with original image
            base_img = self.image.convert('RGBA')
            preview_img = Image.alpha_composite(base_img, overlay)
            self.preview_photo = ImageTk.PhotoImage(preview_img)
            
            # Update canvas
            if hasattr(self, 'canvas_image') and self.canvas_image:
                self.canvas.itemconfig(self.canvas_image, image=self.preview_photo)
            else:
                self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)
            
            food_status = "ON food" if self.white_is_food.get() else "OFF food"
            color_name = "Green" if self.white_is_food.get() else "Red"
            
        except Exception as e:
            print(f"Error in show_mask_preview: {e}")
                
    def start_drawing(self, event):
        if self.photo is None:
            return
        
        self.drawing = True
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.points = [(canvas_x, canvas_y)]
        self.status_var.set("Drawing boundary... Release mouse button when finished")
        
    def draw_line(self, event):
        if not self.drawing or self.photo is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.points:
            # Draw line from last point to current point
            last_x, last_y = self.points[-1]
            self.canvas.create_line(last_x, last_y, canvas_x, canvas_y, 
                                  fill="red", width=2, tags="trace")
            
        self.points.append((canvas_x, canvas_y))
        
        # Update canvas immediately for responsive drawing
        self.canvas.update_idletasks()
        
    def stop_drawing(self, event):
        if not self.drawing:
            return
            
        self.drawing = False
        
        if len(self.points) > 2:
            # Close the polygon by connecting last point to first
            first_x, first_y = self.points[0]
            last_x, last_y = self.points[-1]
            self.canvas.create_line(last_x, last_y, first_x, first_y, 
                                  fill="red", width=2, tags="trace")
            
            self.status_var.set(f"Boundary traced with {len(self.points)} points. Click 'Create Mask' to generate mask.")
        else:
            self.status_var.set("Trace too short. Try again with a longer boundary.")
            
    def clear_trace(self):
        self.canvas.delete("trace")
        self.points = []
        self.mask_array = None
        if self.photo:
            self.status_var.set("Trace cleared. Ready to draw new boundary.")
        
    def create_mask(self):
        if not self.points or len(self.points) < 3:
            messagebox.showwarning("Warning", "Please trace a boundary first!")
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
            
        try:
            # Convert canvas coordinates back to original image coordinates
            original_points = []
            for x, y in self.points:
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)
                original_points.append((orig_x, orig_y))
            
            # Create mask using PIL
            mask_img = Image.new('L', self.original_image.size, 0)  # Black background
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(original_points, fill=255)  # White interior
            
            # Convert to numpy array
            self.mask_array = np.array(mask_img)
            
            # Show preview of mask on canvas
            self.show_mask_preview()
            
            food_status = "ON food" if self.white_is_food.get() else "OFF food"
            color_name = "Green" if self.white_is_food.get() else "Red"
            self.status_var.set(f"Mask created! {color_name} area shows {food_status} region.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create mask: {str(e)}")
    
    def update_food_interpretation(self):
        """Update the mask preview when food interpretation changes"""
        if self.mask_array is not None and self.original_image is not None:
            self.show_mask_preview()  # Use the new preview function
            
    def save_mask(self):
        if self.mask_array is None:
            messagebox.showwarning("Warning", "Please create a mask first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save mask as image
                mask_img = Image.fromarray(self.mask_array, mode='L')
                mask_img.save(file_path)
                
                self.status_var.set(f"Mask saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Mask saved successfully!\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load CSV
                self.csv_data = pd.read_csv(file_path)
                self.csv_file_path = file_path
                
                # Analyze CSV structure
                self.analyze_csv()
                
                self.status_var.set(f"CSV loaded: {os.path.basename(file_path)} ({len(self.csv_data)} rows)")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def analyze_csv(self):
        if self.csv_data is None:
            return
        
        # Clear previous info
        self.csv_info.config(state=tk.NORMAL)
        self.csv_info.delete(1.0, tk.END)
        
        # Analyze CSV structure
        info_text = f"CSV File Analysis:\n"
        info_text += f"Rows: {len(self.csv_data)}\n"
        info_text += f"Columns: {len(self.csv_data.columns)}\n\n"
        
        # Find tracking columns
        worm_ids = set()
        coordinate_cols = []
        
        for col in self.csv_data.columns:
            if col.lower() != 'frame' and ('_x' in col or '_y' in col):
                coordinate_cols.append(col)
                # Extract worm ID
                parts = col.split('_')
                if len(parts) >= 2:
                    worm_id = '_'.join(parts[:-1])  # Everything except the last part (x/y)
                    worm_ids.add(worm_id)
        
        # Sort worm IDs naturally (handling numbers properly)
        def natural_sort_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]
        
        self.worm_ids = sorted(worm_ids, key=natural_sort_key)
        
        info_text += f"Detected {len(self.worm_ids)} tracked objects:\n"
        for worm_id in self.worm_ids:
            info_text += f"  - {worm_id}\n"
        
        info_text += f"\nCoordinate columns: {len(coordinate_cols)}\n"
        
        # Sample data
        info_text += f"\nSample data (first 3 rows):\n"
        for i in range(min(3, len(self.csv_data))):
            row = self.csv_data.iloc[i]
            info_text += f"Frame {row.get('frame', i)}: "
            # Show first few coordinates
            coords_shown = 0
            for col in coordinate_cols[:6]:  # Show first 6 coordinates
                val = row[col]
                if pd.notna(val):
                    info_text += f"{col}={val:.1f} "
                    coords_shown += 1
            if len(coordinate_cols) > 6:
                info_text += "..."
            info_text += "\n"
        
        self.csv_info.insert(tk.END, info_text)
        self.csv_info.config(state=tk.DISABLED)
    
    def process_csv(self):
        if self.csv_data is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        if self.mask_array is None:
            messagebox.showwarning("Warning", "Please create a mask first!")
            return
        
        try:
            # Clear results
            self.results_info.config(state=tk.NORMAL)
            self.results_info.delete(1.0, tk.END)
            
            # Start with original data and insert food_encounter columns in the right places
            processed_data = self.csv_data.copy()
            results_text = "Processing Results:\n\n"
            
            # Process each worm and insert on_food columns
            for worm_id in self.worm_ids:
                x_col = f"{worm_id}_x"
                y_col = f"{worm_id}_y"
                food_col = f"{worm_id}_on_food"
                
                if x_col in self.csv_data.columns and y_col in self.csv_data.columns:
                    # Process coordinates
                    on_food_values = []
                    valid_points = 0
                    food_count = 0
                    
                    for idx, row in self.csv_data.iterrows():
                        x = row[x_col]
                        y = row[y_col]
                        
                        if pd.notna(x) and pd.notna(y):
                            valid_points += 1
                            # Check if point is in mask
                            x_int = int(round(x))
                            y_int = int(round(y))
                            
                            # Bounds check
                            if (0 <= x_int < self.mask_array.shape[1] and 
                                0 <= y_int < self.mask_array.shape[0]):
                                mask_value = self.mask_array[y_int, x_int]
                                
                                # Determine if on food based on mask value and interpretation
                                if self.white_is_food.get():
                                    on_food = mask_value > 128  # White areas are food
                                else:
                                    on_food = mask_value <= 128  # Black areas are food
                                
                                on_food_values.append(1 if on_food else 0)
                                if on_food:
                                    food_count += 1
                            else:
                                # Point outside image bounds
                                on_food_values.append(0)
                        else:
                            # Missing coordinates - use empty value to match input format
                            on_food_values.append(np.nan)
                    
                    # Add the new column to processed_data
                    processed_data[food_col] = on_food_values
                    
                    # Calculate statistics
                    if valid_points > 0:
                        food_percentage = (food_count / valid_points) * 100
                        results_text += f"{worm_id}:\n"
                        results_text += f"  Valid points: {valid_points}\n"
                        results_text += f"  On food: {food_count} ({food_percentage:.1f}%)\n"
                        results_text += f"  Off food: {valid_points - food_count} ({100-food_percentage:.1f}%)\n\n"
            
            # Reorder columns to match desired format: worm1_x, worm1_y, worm1_on_food, worm2_x, worm2_y, worm2_on_food, etc.
            ordered_columns = []
            
            # Add frame column first if it exists
            if 'frame' in processed_data.columns:
                ordered_columns.append('frame')
            
            # Add grouped worm columns
            for worm_id in self.worm_ids:
                x_col = f"{worm_id}_x"
                y_col = f"{worm_id}_y"
                food_col = f"{worm_id}_on_food"
                
                if x_col in processed_data.columns:
                    ordered_columns.append(x_col)
                if y_col in processed_data.columns:
                    ordered_columns.append(y_col)
                if food_col in processed_data.columns:
                    ordered_columns.append(food_col)
            
            # Add any remaining columns that weren't processed
            for col in processed_data.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            # Reorder the dataframe
            processed_data = processed_data[ordered_columns]
            
            # Store processed data
            self.processed_data = processed_data
            
            # Update results display
            food_status = "White=ON food" if self.white_is_food.get() else "White=OFF food"
            results_text += f"Mask interpretation: {food_status}\n"
            results_text += f"Total frames processed: {len(processed_data)}\n"
            results_text += f"On_food columns added: {len(self.worm_ids)}\n"
            results_text += f"Column order: grouped by worm (x, y, on_food)\n"
            
            self.results_info.insert(tk.END, results_text)
            self.results_info.config(state=tk.DISABLED)
            
            self.status_var.set("CSV processed successfully! Click 'Save Results' to export.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process CSV: {str(e)}")
    
    def save_results(self):
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            messagebox.showwarning("Warning", "Please process the CSV first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.processed_data.to_csv(file_path, index=False)
                self.status_var.set(f"Results saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Processed CSV saved successfully!\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    app = BoundaryTracer(root)
    root.mainloop()

if __name__ == "__main__":
    main()