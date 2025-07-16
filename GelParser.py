#1kb+ bands:
# 10000,8000,6000,5000,4000,3000,2000,1500,1200,1000
#!/usr/bin/env python3
"""
Gel Electrophoresis Analysis Tool - Command Line Version

This script provides all the functionality of the notebook version but is more
reliable for interactive use. Run it from the command line with:

python gel_analysis.py
"""

import json
import os
import csv
import uuid
import warnings
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from skimage import io

# Configure environment
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

class GelAnalyzer:
    """class for analyzing gel electrophoresis images"""
    
    def __init__(self):
        """initialize the gel analyzer with default settings"""
        self.image = None
        self.metadata = {
            'experiment_id': str(uuid.uuid4()),
            'date_analyzed': datetime.now().isoformat(),
            'ladder_info': {'bands': []},
            'sample_info': {}
        }
        self.lanes = {}
        self.results = {
            'gel_dimensions': {'y_start': None, 'y_end': None}
        }
        self.band_counter = 1
        self.output_dir = "results"
        self.raw_data = defaultdict(list)
        self.final_data = []
        self.used_band_ids = set()

    def get_next_band_id(self) -> str:
        """generate sequential band IDs"""
        while True:
            new_id = f"B{self.band_counter}"
            if new_id not in self.used_band_ids:
                self.used_band_ids.add(new_id)
                self.band_counter += 1
                return new_id
            self.band_counter += 1

    def initialize_output(self) -> None:
        """create output directory if missing"""
        os.makedirs(self.output_dir, exist_ok=True)

    def save_raw_data(self, lane_num: int) -> None:
        """save raw band data for a lane to CSV file"""
        lane = self.lanes[lane_num]
        current_bands = [
            {'band_id': bid, 'position': pos, 'lane': lane_num}
            for bid, pos in zip(lane['band_ids'], lane['bands'])
        ]
        
        lane_csv_path = f"{self.output_dir}/lane_{lane_num}_raw.csv"
        with open(lane_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['band_id', 'position', 'lane'])
            writer.writeheader()
            writer.writerows(current_bands)

    def load_raw_data(self) -> None:
        """load raw data from CSV files"""
        max_id = 0
        for lane_num in self.lanes:
            lane_csv_path = f"{self.output_dir}/lane_{lane_num}_raw.csv"
            if os.path.exists(lane_csv_path):
                with open(lane_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    bands = []
                    band_ids = []
                    for row in reader:
                        bands.append(int(row['position']))
                        bid = row['band_id']
                        band_ids.append(bid)
                        if bid.startswith('B'):
                            try:
                                current_id = int(bid[1:])
                                if current_id > max_id:
                                    max_id = current_id
                            except ValueError:
                                pass
                
                self.lanes[lane_num]['bands'] = bands
                self.lanes[lane_num]['band_ids'] = band_ids
        
        if max_id > 0:
            self.band_counter = max_id + 1
            for lane in self.lanes.values():
                self.used_band_ids.update(lane['band_ids'])

    def load_image(self, file_path: str) -> bool:
        """load and normalize image"""
        try:
            self.image = io.imread(file_path)
            if len(self.image.shape) == 3:  # Convert to grayscale if RGB
                self.image = np.mean(self.image, axis=2)
            self.image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
            self.metadata['image_file'] = os.path.basename(file_path)
            self.metadata['image_dimensions'] = self.image.shape
            self.initialize_output()
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False

    def set_gel_dimensions(self) -> None:
        """interactive selection of gel dimensions"""
        print("\n=== Gel Dimensions Selection ===")
        print("Please select two points on the image:")
        print("1. First click at the top of the wells (y0)")
        print("2. Second click at the bottom of the gel (yn)")
        
        fig, ax = plt.subplots(figsize=(12,8))
        ax.imshow(self.image, cmap='gray')
        ax.set_title("Click first on well (y0), then on gel bottom (yn)")
        
        points = []
        def onclick(event):
            if event.inaxes != ax:
                return
                
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            
            if len(points) == 1:
                ax.set_title("Now click on gel bottom")
                plt.draw()
            elif len(points) == 2:
                plt.close()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)  # This makes the plot stay open until closed
        
        if len(points) == 2:
            self.results['gel_dimensions']['y_start'] = int(points[0][1])
            self.results['gel_dimensions']['y_end'] = int(points[1][1])
            print(f"\nGel dimensions set: y0={points[0][1]}, yn={points[1][1]}")
        else:
            raise ValueError("Exactly two points must be selected")

    def detect_lanes(self, lane_width: int = 30, min_lanes: int = 7) -> int:
        """automatically detect lanes in the image"""
        try:
            profile = np.mean(self.image, axis=0)
            smoothed = savgol_filter(profile, 101, 3)
            
            min_prominence = 0.4 * (np.max(smoothed) - np.min(smoothed))
            peaks, _ = find_peaks(-smoothed, prominence=min_prominence, distance=lane_width*0.8)
            
            if len(peaks) >= min_lanes-1:
                lane_centers = np.sort(peaks)
            else:
                lane_centers = np.linspace(lane_width, self.image.shape[1]-lane_width, min_lanes)
            
            self.lanes = {}
            for i, center in enumerate(lane_centers):
                self.lanes[i+1] = {
                    'x_center': center,
                    'x_start': max(0, int(center - lane_width//2)),
                    'x_end': min(self.image.shape[1]-1, int(center + lane_width//2)),
                    'profile': None,
                    'bands': [],
                    'band_ids': [],
                    'rf_values': [],
                    'estimated_sizes': []
                }
                
                lane_img = self.image[:, self.lanes[i+1]['x_start']:self.lanes[i+1]['x_end']]
                self.lanes[i+1]['profile'] = cv2.GaussianBlur(
                    np.mean(lane_img, axis=1).astype(np.float32).reshape(-1,1), 
                    (71,1), 0).flatten()
            
            print(f"Successfully detected {len(self.lanes)} lanes")
            return len(self.lanes)
            
        except Exception as e:
            print(f"Automatic detection failed: {str(e)}")
            return self.create_fallback_lanes(min_lanes, lane_width)

    def create_fallback_lanes(self, num_lanes: int, lane_width: int) -> int:
        """create evenly spaced lanes when automatic detection fails"""
        step = (self.image.shape[1] - lane_width) / (num_lanes - 1)
        for i in range(num_lanes):
            center = lane_width//2 + i * step
            self.lanes[i+1] = {
                'x_center': center,
                'x_start': int(center - lane_width//2),
                'x_end': int(center + lane_width//2),
                'profile': None,
                'bands': [],
                'band_ids': [],
                'rf_values': [],
                'estimated_sizes': []
            }
            lane_img = self.image[:, self.lanes[i+1]['x_start']:self.lanes[i+1]['x_end']]
            self.lanes[i+1]['profile'] = np.mean(lane_img, axis=1)
        print(f"Fallback: Created {num_lanes} evenly spaced lanes")
        return num_lanes

    def plot_lane_detection(self) -> None:
        """visualize detected lanes"""
        plt.figure(figsize=(15,5))
        plt.imshow(self.image, cmap='gray')
        for lane_num, lane in self.lanes.items():
            plt.axvline(lane['x_start'], color='cyan', alpha=0.5, linestyle='--')
            plt.axvline(lane['x_end'], color='cyan', alpha=0.5, linestyle='--')
            plt.text(lane['x_center'], 20, str(lane_num), 
                    color='yellow', ha='center', va='top', fontsize=12, 
                    bbox=dict(facecolor='black', alpha=0.5))
        plt.title("Detected Lanes with Bandwidth Markers")
        plt.show(block=True)

    def detect_bands(self, lane_num: int, top_cut: int = 0, 
                    min_prominence: float = 10, min_distance: int = 5) -> List[int]:
        """detect bands in a specific lane using peak finding."""
        try:
            profile = self.lanes[lane_num]['profile'][top_cut:]
            peaks, properties = find_peaks(
                profile, 
                prominence=min_prominence,
                width=2,
                distance=min_distance,
                rel_height=0.9
            )
            peaks += top_cut
            
            band_ids = [self.get_next_band_id() for _ in range(len(peaks))]
            
            self.lanes[lane_num]['bands'] = sorted(peaks.tolist())
            self.lanes[lane_num]['band_ids'] = band_ids
            
            plt.figure(figsize=(12,5))
            plt.plot(self.lanes[lane_num]['profile'], label='Intensity profile', linewidth=1)
            for i, (peak, bid) in enumerate(zip(peaks, band_ids)):
                plt.plot(peak, self.lanes[lane_num]['profile'][peak], 
                        '+', markersize=12, markeredgewidth=2, label=f'Band {bid}')
            plt.gca().invert_yaxis()
            plt.title(f"Lane {lane_num} - Band Detection")
            plt.legend()
            plt.show(block=True)
            
            self.save_raw_data(lane_num)
            
            return peaks
        except Exception as e:
            print(f"Error detecting bands in Lane {lane_num}: {str(e)}")
            return []

    def interactive_band_editor(self, lane_num: int) -> None:
        """provide interactive editor for adding/removing bands"""
        print("\n=== Interactive Band Editor ===")
        print("Left click (+) to add a band")
        print("Right click to remove a band")
        print("Close the window when finished")
        
        fig, ax = plt.subplots(figsize=(16,10))
        ax.imshow(self.image, cmap='gray')
        
        x = self.lanes[lane_num]['x_center']
        for bid, pos in zip(self.lanes[lane_num]['band_ids'], self.lanes[lane_num]['bands']):
            ax.plot(x, pos, '+', color='cyan', markersize=12, markeredgewidth=2)
            ax.text(x+8, pos, bid, color='cyan', fontsize=10, 
                   bbox=dict(facecolor='black', alpha=0.5))
        
        def onclick(event):
            if event.inaxes != ax:
                return
                
            y_pos = int(event.ydata)
            
            if event.button == 1:  # Left click: Add band
                if all(abs(pos - y_pos) > 8 for pos in self.lanes[lane_num]['bands']):
                    bid = self.get_next_band_id()
                    self.lanes[lane_num]['bands'].append(y_pos)
                    self.lanes[lane_num]['band_ids'].append(bid)
                    sorted_bands = sorted(zip(self.lanes[lane_num]['bands'], self.lanes[lane_num]['band_ids']))
                    self.lanes[lane_num]['bands'], self.lanes[lane_num]['band_ids'] = zip(*sorted_bands)
                    self.lanes[lane_num]['bands'] = list(self.lanes[lane_num]['bands'])
                    self.lanes[lane_num]['band_ids'] = list(self.lanes[lane_num]['band_ids'])
                    print(f"Added band {bid} at y={y_pos} in Lane {lane_num}")
                else:
                    print("Warning: Too close to existing band! Minimum distance: 8 pixels")
            
            elif event.button == 3:  # Right click: Remove band
                closest_idx = np.argmin([abs(pos - y_pos) for pos in self.lanes[lane_num]['bands']])
                if abs(self.lanes[lane_num]['bands'][closest_idx] - y_pos) < 15:
                    removed_bid = self.lanes[lane_num]['band_ids'].pop(closest_idx)
                    removed_pos = self.lanes[lane_num]['bands'].pop(closest_idx)
                    self.used_band_ids.discard(removed_bid)
                    print(f"Removed band {removed_bid} at y={removed_pos} from Lane {lane_num}")
            
            ax.clear()
            ax.imshow(self.image, cmap='gray')
            for bid, pos in zip(self.lanes[lane_num]['band_ids'], self.lanes[lane_num]['bands']):
                ax.plot(x, pos, '+', color='cyan', markersize=12, markeredgewidth=2)
                ax.text(x+8, pos, bid, color='cyan', fontsize=10, 
                       bbox=dict(facecolor='black', alpha=0.5))
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.title(f"Lane {lane_num} Editor: Left click (+) to add | Right click to remove")
        plt.show(block=True)
        
        self.save_raw_data(lane_num)

    def calculate_rf(self, ladder_lane: int) -> None:
        """calculate Rf values for all bands relative to ladder fit"""
        self.load_raw_data()
        
        if not self.metadata['ladder_info'].get('bands'):
            raise ValueError("No ladder bands defined.")
        
        if not self.results['gel_dimensions']['y_start'] or not self.results['gel_dimensions']['y_end']:
            raise ValueError("Gel dimensions not set. Please set y0 and yn first.")
        
        y0 = self.results['gel_dimensions']['y_start']
        yn = self.results['gel_dimensions']['y_end']
        
        # Calculate Rf for all bands in all lanes
        for lane_num, lane in self.lanes.items():
            lane['rf_values'] = [
                (pos - y0) / (yn - y0)
                for pos in lane['bands']
            ]
            
            # Store ladder sizes directly from metadata
            if lane_num == ladder_lane:
                lane['estimated_sizes'] = self.metadata['ladder_info']['bands'][:len(lane['bands'])]
            else:
                lane['estimated_sizes'] = []  # Will be calculated later during estimation

    def fit_linear_calibration(self, ladder_lane: int) -> None:
        """fit linear calibration curve using ladder bands."""
        # Get ladder Rf values and known sizes
        x = np.array(self.lanes[ladder_lane]['rf_values'])
        y = np.log10(np.array(self.lanes[ladder_lane]['estimated_sizes']))
        
        # Linear regression in log space
        def linear_func(x, a, b):
            return a * x + b
        
        popt, _ = curve_fit(linear_func, x, y)
        
        # Calculate R²
        y_pred = linear_func(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Store calibration parameters
        self.results['calibration'] = {
            'function': 'log10(bp) = a*Rf + b',
            'params': popt.tolist(),
            'r_squared': r_squared,
            'ladder_lane': ladder_lane,
            'used_bands': len(x),
            'equation': f"log10(bp) = {popt[0]:.4f}*Rf + {popt[1]:.4f}"
        }
        
        print(f"\nCalibration successful:")
        print(f"log10(bp) = {popt[0]:.4f}*Rf + {popt[1]:.4f}")
        print(f"R² = {r_squared:.4f} (based on {len(x)} ladder bands)")

    def estimate_sizes(self) -> None:
        """estimate fragment sizes for all bands using calibration."""
        if 'calibration' not in self.results:
            raise ValueError("Calibration must be performed first!")
        
        a, b = self.results['calibration']['params']
        ladder_lane = self.results['calibration']['ladder_lane']
        
        # Calculate sizes for all bands in all lanes (including ladder for verification)
        for lane_num, lane in self.lanes.items():
            lane['estimated_sizes'] = [10**(a * rf + b) for rf in lane['rf_values']]
        
        # Overwrite ladder sizes with known values for consistency
        self.lanes[ladder_lane]['estimated_sizes'] = self.metadata['ladder_info']['bands'][:len(self.lanes[ladder_lane]['bands'])]
        
        self.prepare_final_data()

    def prepare_final_data(self) -> None:
        """prepare final data structure for export"""
        self.final_data = []
        ladder_lane = self.results.get('calibration', {}).get('ladder_lane', -1)
        
        for lane_num, lane in self.lanes.items():
            for bid, pos, size, rf in zip(
                lane['band_ids'],
                lane['bands'],
                lane.get('estimated_sizes', []),
                lane.get('rf_values', [])
            ):
                self.final_data.append({
                    'band_id': bid,
                    'lane': lane_num,
                    'position_px': pos,
                    'size_bp': size if lane_num != ladder_lane else int(size),
                    'rf_value': rf,
                    'is_ladder': lane_num == ladder_lane
                })

    def visualize_results(self) -> plt.Figure:
        """visualization of results"""
        fig = plt.figure(figsize=(20, 14))
        
        # Main gel image
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        ax1.imshow(self.image, cmap='gray')
        
        # Calibration curve
        ax2 = plt.subplot2grid((3, 2), (2, 0), colspan=1, rowspan=1)
        
        # Band table
        ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=1, rowspan=1)
        ax3.axis('off')
        
        ladder_lane = self.results.get('calibration', {}).get('ladder_lane', -1)
        
        # Plot all bands
        for lane_num, lane in self.lanes.items():
            x = lane['x_center']
            color = 'yellow' if lane_num == ladder_lane else 'cyan'
            
            for bid, pos, size in zip(lane['band_ids'], lane['bands'], lane['estimated_sizes']):
                ax1.plot(x, pos, '+', color=color, markersize=12, markeredgewidth=2)
                label = f"{bid}\n{size:.0f}bp" if lane_num != ladder_lane else f"{bid}\n{size}bp"
                ax1.text(x+8, pos, label, color=color, fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5))
        
        # Add metadata and calibration info
        metadata_text = (
            f"Analyst: {self.metadata.get('analyst', 'N/A')}\n"
            f"Date: {self.metadata.get('experiment_date', 'N/A')}\n"
            f"Gel: {self.metadata.get('gel_type', 'N/A')}% agarose\n"
            f"Voltage: {self.metadata.get('voltage', 'N/A')}V\n"
            f"Run time: {self.metadata.get('run_time', 'N/A')} min"
        )
        
        calib_text = ""
        if 'calibration' in self.results:
            calib = self.results['calibration']
            calib_text = (
                f"Calibration: {calib.get('equation', 'N/A')}\n"
                f"R² = {calib.get('r_squared', 0):.4f}\n"
                f"Ladder lane: {calib.get('ladder_lane', 'N/A')}"
            )
        
        ax1.text(0.02, 0.98, metadata_text, 
                transform=ax1.transAxes, color='white', 
                fontsize=10, va='top', ha='left',
                bbox=dict(facecolor='black', alpha=0.7))
        
        ax1.text(0.98, 0.98, calib_text, 
                transform=ax1.transAxes, color='white', 
                fontsize=10, va='top', ha='right',
                bbox=dict(facecolor='black', alpha=0.7))
        
        ax1.set_title("Gel Image with All Bands (Yellow=Ladder, Cyan=Samples)")

        # Plot calibration curve
        if 'calibration' in self.results:
            calib = self.results['calibration']
            x = np.linspace(0, 1, 100)
            y = 10**(calib['params'][0] * x + calib['params'][1])
            
            ax2.set_yscale('log')
            ax2.plot(x, y, label='Calibration curve')
            
            if ladder_lane in self.lanes:
                rf_values = self.lanes[ladder_lane]['rf_values']
                bp_sizes = self.lanes[ladder_lane]['estimated_sizes']
                ax2.plot(rf_values, bp_sizes, 'ro', label='Ladder Bands')
            
            ax2.set_xlabel("Rf Value")
            ax2.set_ylabel("Fragment Size (bp)")
            ax2.legend()
            ax2.set_title(f"Linear Calibration (R²={calib['r_squared']:.4f})")
        
        # Create band table
        table_data = []
        for lane_num, lane in self.lanes.items():
            for bid, pos, size, rf in zip(
                lane['band_ids'],
                lane['bands'],
                lane['estimated_sizes'],
                lane['rf_values']
            ):
                table_data.append([
                    bid,
                    lane_num,
                    pos,
                    f"{size:.0f}" if lane_num != ladder_lane else f"{size}",
                    f"{rf:.3f}"
                ])
        
        if table_data:
            ax3.table(
                cellText=table_data,
                colLabels=['Band ID', 'Lane', 'Position (px)', 'Size (bp)', 'Rf Value'],
                loc='center',
                cellLoc='center'
            )
            ax3.set_title("Band Overview (*exact ladder values)")
        
        plt.tight_layout()
        plt.show(block=True)
        return fig

    def export_results(self) -> bool:
        """export all results to files"""
        try:
            base_name = f"gel_{self.metadata['experiment_id']}"
            ladder_lane = self.results.get('calibration', {}).get('ladder_lane', -1)
            
            # 1. Final CSV
            final_csv_path = f"{self.output_dir}/{base_name}_final.csv"
            pd.DataFrame(self.final_data).to_csv(final_csv_path, index=False)
            
            # 2. JSON Export
            export_data = {
                'metadata': self.metadata,
                'results': self.results,
                'lanes': {
                    str(lane_num): {
                        'position': {
                            'center': lane['x_center'],
                            'start': lane['x_start'],
                            'end': lane['x_end']
                        },
                        'bands': [
                            {
                                'id': bid,
                                'position_px': pos,
                                'size_bp': size if lane_num != ladder_lane else int(size),
                                'rf_value': rf
                            }
                            for bid, pos, size, rf in zip(
                                lane['band_ids'],
                                lane['bands'],
                                lane.get('estimated_sizes', []),
                                lane.get('rf_values', [])
                            )
                        ]
                    }
                    for lane_num, lane in self.lanes.items()
                }
            }
            
            json_path = f"{self.output_dir}/{base_name}_full.json"
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # 3. Save visualization
            fig = self.visualize_results()
            png_path = f"{self.output_dir}/{base_name}_plot.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='w')
            plt.close(fig)
            
            print(f"\nFinal results exported to:")
            print(f"- {os.path.abspath(final_csv_path)}")
            print(f"- {os.path.abspath(json_path)}")
            print(f"- {os.path.abspath(png_path)}")
            
            return True
        except Exception as e:
            print(f"\nError exporting results: {str(e)}")
            return False

def main():
    """Und ab gehts"""
    print("\n=== Gel Electrophoresis Analysis Tool ===")
    print("This tool will guide you through analyzing your gel image.\n")
    
    analyzer = GelAnalyzer()
    
    # Load image
    while True:
        file_path = input("Path to gel image: ").strip()
        if analyzer.load_image(file_path):
            break
        print("Please try again with a valid image file path.")
    
    # Collect metadata
    print("\n=== Experiment Metadata ===")
    analyzer.metadata['analyst'] = input("Analyst name: ").strip()
    analyzer.metadata['experiment_date'] = input("Date (YYYY-MM-DD): ").strip()
    analyzer.metadata['gel_type'] = input("Agarose concentration (%): ").strip()
    analyzer.metadata['voltage'] = input("Run voltage (V): ").strip()
    analyzer.metadata['run_time'] = input("Run time (minutes): ").strip()

    print("\n=== Ladder Information ===")
    analyzer.metadata['ladder_info']['type'] = input("Ladder type (e.g., '100bp', '1kb'): ").strip()
    while True:
        known_bp = input("Known bp values (comma separated, top to bottom): ").strip()
        try:
            analyzer.metadata['ladder_info']['bands'] = [int(bp) for bp in known_bp.split(',')]
            break
        except ValueError:
            print("Please enter only numbers separated by commas (e.g., '1000,500,250')")

    print("\n=== Sample Information ===")
    while True:
        try:
            num_samples = int(input("Number of samples (excluding ladder): ").strip())
            break
        except ValueError:
            print("Please enter a number.")
    
    for i in range(num_samples):
        sample_id = input(f"\nSample ID {i+1}: ").strip()
        description = input(f"Description for {sample_id}: ").strip()
        analyzer.metadata['sample_info'][sample_id] = {
            'lane': i+2,
            'description': description
        }
    
    # Set gel dimensions
    print("\n=== Gel Dimensions ===")
    while True:
        try:
            analyzer.set_gel_dimensions()
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.")
    
    # Detect lanes
    print("\n=== Lane Detection ===")
    while True:
        try:
            lane_width = int(input("Estimated lane width in pixels (20-50) [30]: ").strip() or 30)
            min_lanes = int(input("Minimum number of lanes (incl. ladder) [7]: ").strip() or 7)
            
            detected = analyzer.detect_lanes(lane_width=lane_width, min_lanes=min_lanes)
            analyzer.plot_lane_detection()
            
            if detected >= min_lanes or input(f"Only {detected} lanes detected. Continue? (y/n): ").lower() == 'y':
                break
        except ValueError:
            print("Please enter integers!")
    
    # Identify ladder lane
    print("\n=== Ladder Lane ===")
    print(f"Available lanes: {list(analyzer.lanes.keys())}")
    while True:
        try:
            ladder_lane = int(input("Which lane is the ladder? (enter number): ").strip())
            if ladder_lane in analyzer.lanes:
                break
            print(f"Invalid lane number. Available lanes: {list(analyzer.lanes.keys())}")
        except ValueError:
            print("Please enter a number!")
    
    # Analyze ladder bands
    print("\n=== Analyze Ladder Bands ===")
    while True:
        try:
            analyzer.detect_bands(ladder_lane, min_distance=5)
            analyzer.interactive_band_editor(ladder_lane)
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.")
    
    # Perform calibration
    analyzer.calculate_rf(ladder_lane)
    analyzer.fit_linear_calibration(ladder_lane)
    
    # Analyze sample bands
    print("\n=== Analyze Sample Bands ===")
    for lane_num in analyzer.lanes:
        if lane_num != ladder_lane:
            print(f"\nLane {lane_num}:")
            while True:
                try:
                    analyzer.detect_bands(lane_num, min_distance=5)
                    analyzer.interactive_band_editor(lane_num)
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print("Please try again.")
    
    # Estimate sizes and export
    analyzer.estimate_sizes()
    
    if not analyzer.export_results():
        print("\nAnalysis completed with export errors")
    else:
        print("\nAnalysis successfully completed!")

if __name__ == "__main__":
    main()